/**
 * Inverted index for sparse vector similarity search.
 *
 * Sparse vectors represent documents as (term_id, weight) pairs - most
 * entries are zero.  To find the top-K documents most similar to a query
 * vector we need to compute dot products:
 *
 *  score(doc) = sum over matching terms of (query_weight * doc_weight)
 *
 * The search algorithm processes documents in contiguous ID batches.  For each
 * batch it accumulates partial dot-product scores into a flat buffer, then
 * pushes the best scores into a min-heap of size K.  A lightweight pruning
 * step can skip large sections of the longest posting list when their upper
 * bound cannot beat the current K-th best score.
 *
 * Storage uses MDBX.  Each term's posting list is stored as a single key-value
 * pair:
 *     key   = term_id  (uint32)
 *     value = PostingListHeader | doc_ids[n] (uint32) | values[n] (uint8 or float)
 *
 * Values are either stored as raw floats (NDD_INV_IDX_STORE_FLOATS) or
 * quantized to uint8 relative to the list's max value to save space.
 * Deleted entries are tombstoned (value = 0) and periodically compacted.
 */

#include "inverted_index.hpp"
#include "../utils/log.hpp"

namespace ndd {

    InvertedIndex::InvertedIndex(MDBX_env* env, size_t vocab_size)
        : env_(env), vocab_size_(vocab_size) {}

    bool InvertedIndex::initialize() {
        std::unique_lock<std::shared_mutex> lock(mutex_);

        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_READWRITE, &txn);
        if (rc != 0) {
            LOG_ERROR("Failed to begin init transaction: " << mdbx_strerror(rc));
            return false;
        }

        rc = mdbx_dbi_open(txn, "term_postings",
                            MDBX_CREATE | MDBX_INTEGERKEY, &term_postings_dbi_);
        if (rc != 0) {
            LOG_ERROR("Failed to open term_postings dbi: " << mdbx_strerror(rc));
            mdbx_txn_abort(txn);
            return false;
        }

        rc = mdbx_txn_commit(txn);
        if (rc != 0) {
            LOG_ERROR("Failed to commit init transaction: " << mdbx_strerror(rc));
            return false;
        }

        return loadTermInfo();
    }

    bool InvertedIndex::addDocumentsBatch(
        MDBX_txn* txn,
        const std::vector<std::pair<ndd::idInt, SparseVector>>& docs)
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        return addDocumentsBatchInternal(txn, docs);
    }

    bool InvertedIndex::removeDocument(MDBX_txn* txn,
                                    ndd::idInt doc_id,
                                    const SparseVector& vec)
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        return removeDocumentInternal(txn, doc_id, vec);
    }

    size_t InvertedIndex::getTermCount() const {
        return term_info_.size();
    }

    size_t InvertedIndex::getVocabSize() const {
        return vocab_size_;
    }

    /**
     * =====================================================================
     * Search - batched dot-product accumulation with pruning
     * =====================================================================
     *
     * Algorithm overview:
     *
     * 1. For each query term, open a read-only iterator over its
     *    posting list (zero-copy pointer into the MDBX page).
     *
     * 2. Process document IDs in contiguous batches of settings::INV_IDX_SEARCH_BATCH_SZ.
     *    - For each posting list, walk entries whose doc_id falls
     *      within [batch_start, batch_end].
     *    - Accumulate: scores_buf[doc_id - batch_start] += doc_weight * query_weight
     *
     * 3. After accumulating, scan the buffer for scores that beat the
     *    current K-th best. Push winners into a min-heap of size K.
     *
     * 4. Between batches, try to prune the longest posting list: if
     *    its maximum possible contribution (global_max * query_weight)
     *    cannot beat the current K-th best, skip ahead.
     *
     * 5. Repeat until all iterators are exhausted.
     */
    std::vector<std::pair<ndd::idInt, float>>
    InvertedIndex::search(const SparseVector& query,
                            size_t k,
                            const ndd::RoaringBitmap* filter)
    {
        if (query.empty() || k == 0) {
            return {};
        }

        std::shared_lock<std::shared_mutex> lock(mutex_);

        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn);
        if (rc != 0) {
            LOG_ERROR("Failed to begin search transaction: " << mdbx_strerror(rc));
            return {};
        }

        // -- STEP 1: Build an iterator per query term --

        std::vector<PostingListIterator> iters_storage;
        std::vector<PostingListIterator*> iters;
        iters_storage.reserve(query.indices.size());
        iters.reserve(query.indices.size());

        for (size_t qi = 0; qi < query.indices.size(); qi++) {
            uint32_t term_id = query.indices[qi];
            float qw = query.values[qi];
            if (qw <= 0.0f) continue;

            auto info_it = term_info_.find(term_id);
            if (info_it == term_info_.end()) continue;

            PostingListView view = getReadOnlyPostingList(txn, term_id);
            if (view.doc_ids == nullptr || view.count == 0) continue;

            PostingListIterator it;
            it.term_id = term_id;
            it.term_weight = qw;
            it.global_max = info_it->second;
            it.index = this;
            it.init(view);

            if (it.current_doc_id != EXHAUSTED_DOC_ID) {
                iters_storage.push_back(it);
            }
        }

        for (size_t i = 0; i < iters_storage.size(); i++) {
            iters.push_back(&iters_storage[i]);
        }

        if (iters.empty()) {
            mdbx_txn_abort(txn);
            return {};
        }

        // -- STEP 2: Prepare scoring state --

        bool use_pruning = (iters.size() > 1);
        float best_min_score = 0.0f;

        std::vector<float> scores_buf(settings::INV_IDX_SEARCH_BATCH_SZ, 0.0f);
        std::priority_queue<ScoredDoc> top_results;  // min-heap of size k
        float threshold = 0.0f;  // score of the current k-th best result

        // Start iterating from the smallest doc_id across all iterators.
        auto minIterDocId = [&iters]() -> ndd::idInt {
            ndd::idInt min_id = EXHAUSTED_DOC_ID;
            for (size_t i = 0; i < iters.size(); i++) {
                if (iters[i]->current_doc_id < min_id) {
                    min_id = iters[i]->current_doc_id;
                }
            }
            return min_id;
        };

        ndd::idInt min_id = minIterDocId();

        // -- STEP 3: Main scoring loop, one batch per iteration --

        while (min_id != EXHAUSTED_DOC_ID) {
            ndd::idInt batch_start = min_id;
            ndd::idInt batch_end = batch_start +
                                (ndd::idInt)settings::INV_IDX_SEARCH_BATCH_SZ - 1;
            if (batch_end < batch_start) {
                batch_end = EXHAUSTED_DOC_ID - 1;  // overflow guard
            }
            size_t batch_len = (size_t)(batch_end - batch_start) + 1;

            if (batch_len > scores_buf.size()) {
                scores_buf.resize(batch_len);
            }
            std::memset(scores_buf.data(), 0, batch_len * sizeof(float));

            /**
             * 3a - Accumulate partial dot-product scores.
             *
             * For each posting list, walk entries with doc_id in
             * [batch_start, batch_end] and add weight * query_weight
             * into the score buffer.
            */
            for (size_t i = 0; i < iters.size(); i++) {
                PostingListIterator* it = iters[i];
                float qw = it->term_weight;

                const uint32_t* ids = it->doc_ids;
                uint32_t idx = it->current_entry_idx;
                uint32_t sz  = it->data_size;

#if defined(NDD_INV_IDX_STORE_FLOATS)
                const float* vals = (const float*)it->values_ptr;
                while (idx < sz && ids[idx] <= batch_end) {
                    if (vals[idx] > 0.0f) {
                        size_t local = (size_t)(ids[idx] - batch_start);
                        scores_buf[local] += vals[idx] * qw;
                    }
                    idx++;
                }
#else
                const uint8_t* vals = (const uint8_t*)it->values_ptr;
                float scale = it->max_value / UINT8_MAX;
                while (idx < sz && ids[idx] <= batch_end) {
                    if (vals[idx] > 0) {
                        size_t local = (size_t)(ids[idx] - batch_start);
                        scores_buf[local] += ((float)vals[idx] * scale) * qw;
                    }
                    idx++;
                }
#endif

                it->current_entry_idx = idx;
                it->advanceToNextLive();
            }

            // 3b - Scan the score buffer and keep the top K.
            for (size_t local = 0; local < batch_len; local++) {
                float s = scores_buf[local];
                if (s == 0.0f || s <= threshold) continue;

                ndd::idInt doc_id = batch_start + (ndd::idInt)local;
                if (filter && !filter->contains(doc_id)) continue;

                if (top_results.size() < k) {
                    top_results.emplace(doc_id, s);
                    if (top_results.size() == k) {
                        threshold = top_results.top().score;
                    }
                } else if (s > threshold) {
                    top_results.pop();
                    top_results.emplace(doc_id, s);
                    threshold = top_results.top().score;
                }
            }

            // 3c - Drop any iterators that have no more entries.
            size_t write_idx = 0;
            for (size_t i = 0; i < iters.size(); i++) {
                if (iters[i]->current_doc_id != EXHAUSTED_DOC_ID) {
                    iters[write_idx++] = iters[i];
                }
            }
            iters.resize(write_idx);
            if (iters.empty()) break;

            // 3d - Find where the next batch starts.
            min_id = minIterDocId();

            // 3e - Pruning.  See pruneLongest() for details.
            if (use_pruning && top_results.size() >= k) {
                float new_min_score = threshold;
                if (new_min_score != best_min_score) {
                    best_min_score = new_min_score;
                    pruneLongest(iters, new_min_score);

                    min_id = minIterDocId();
                }
            }
        }

        // -- STEP 4: Return results in descending score order --

        mdbx_txn_abort(txn);

        std::vector<std::pair<ndd::idInt, float>> results;
        results.reserve(top_results.size());
        while (!top_results.empty()) {
            results.push_back(
                std::make_pair(top_results.top().doc_id, top_results.top().score));
            top_results.pop();
        }
        std::reverse(results.begin(), results.end());
        return results;
    }

    /**
     * =====================================================================
     * Quantization
     * =====================================================================
     *
     * When NDD_INV_IDX_STORE_FLOATS is not defined, weights are stored
     * as uint8 values [0..255] relative to the posting list's max_value.
     * This halves storage compared to float but introduces small rounding.
     *
     * NOTE: quantize(0) returns 0 which doubles as the tombstone marker, so
     * we clamp live entries to at least 1.
     */
    inline uint8_t InvertedIndex::quantize(float val, float max_val) {
        if (max_val <= settings::NEAR_ZERO)
            return 0;
        float scaled = (val / max_val) * UINT8_MAX;

        if (scaled >= UINT8_MAX)
            return UINT8_MAX;

        if (scaled <= 0.0f)
            return 0;

        uint8_t result = (uint8_t)(scaled + 0.5f);
        return result == 0 ? 1 : result;  // keep live entries non-zero
    }

    inline float InvertedIndex::dequantize(uint8_t val, float max_val) {
        if (max_val <= settings::NEAR_ZERO)
            return 0.0f;
        return (float)val * (max_val / UINT8_MAX);
    }


    // =========================================================================
    // SIMD helpers
    // =========================================================================

    size_t InvertedIndex::findDocIdSIMD(const uint32_t* doc_ids,
                                    size_t size,
                                    size_t start_idx,
                                    uint32_t target) const
    {
        size_t idx = start_idx;

#if defined(USE_AVX512)
        const size_t simd_width = 16;
        __m512i target_vec = _mm512_set1_epi32((int)target);

        while (idx + simd_width <= size) {
            __m512i data_vec = _mm512_loadu_si512(doc_ids + idx);
            __mmask16 mask = _mm512_cmpge_epu32_mask(data_vec, target_vec);

            if (mask != 0) {
                return idx + __builtin_ctz(mask);
            }
            idx += simd_width;
        }
#elif defined(USE_AVX2)
        const size_t simd_width = 8;
        __m256i target_vec = _mm256_set1_epi32((int)target);

        while (idx + simd_width <= size) {
            __builtin_prefetch(doc_ids + idx + 32);

            // Quick scalar check: if the last element in this chunk is
            // still below target, skip the whole chunk without loading
            // into a SIMD register.
            if (doc_ids[idx + simd_width - 1] < target) {
                idx += simd_width;
                continue;
            }

            __m256i data_vec =
                _mm256_loadu_si256((const __m256i*)(doc_ids + idx));
            // Unsigned >= via: max(a,b)==a  iff  a >= b
            __m256i max_vec = _mm256_max_epu32(data_vec, target_vec);
            __m256i cmp = _mm256_cmpeq_epi32(max_vec, data_vec);

            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
            if (mask != 0) {
                return idx + __builtin_ctz(mask);
            }
            idx += simd_width;
        }
#elif defined(USE_SVE2)
        svbool_t pg = svwhilelt_b32(idx, size);
        svuint32_t target_vec = svdup_u32(target);

        while (svptest_any(svptrue_b32(), pg)) {
            svuint32_t data_vec = svld1_u32(pg, doc_ids + idx);
            svbool_t cmp = svcmpge_u32(pg, data_vec, target_vec);

            if (svptest_any(pg, cmp)) {
                svbool_t before_match = svbrkb_z(pg, cmp);
                uint64_t count = svcntp_b32(pg, before_match);
                return idx + count;
            }
            idx += svcntw();
            pg = svwhilelt_b32(idx, size);
        }
        return idx;
#elif defined(USE_NEON)
        const size_t simd_width = 4;
        uint32x4_t target_vec = vdupq_n_u32(target);

        while (idx + simd_width <= size) {
            uint32x4_t data_vec = vld1q_u32(doc_ids + idx);
            uint32x4_t cmp = vcgeq_u32(data_vec, target_vec);

            if (vmaxvq_u32(cmp) != 0) {
                for (size_t i = 0; i < simd_width; i++) {
                    if (doc_ids[idx + i] >= target) {
                        return idx + i;
                    }
                }
            }
            idx += simd_width;
        }
#endif

        // Scalar fallback for remaining elements
        while (idx < size && doc_ids[idx] < target) {
            idx++;
        }
        return idx;
    }

    size_t InvertedIndex::findNextLiveSIMD(const uint8_t* values,
                                        size_t size,
                                        size_t start_idx) const
    {
        size_t idx = start_idx;

#if defined(USE_AVX512)
        const size_t simd_width = 64;
        __m512i zero_vec = _mm512_setzero_si512();

        while (idx + simd_width <= size) {
            __m512i data_vec = _mm512_loadu_si512(values + idx);
            __mmask64 mask = _mm512_cmpneq_epu8_mask(data_vec, zero_vec);

            if (mask != 0) {
                return idx + __builtin_ctzll(mask);
            }
            idx += simd_width;
        }
#elif defined(USE_AVX2)
        const size_t simd_width = 32;
        __m256i zero_vec = _mm256_setzero_si256();

        while (idx + simd_width <= size) {
            __m256i data_vec =
                _mm256_loadu_si256((const __m256i*)(values + idx));
            __m256i cmp = _mm256_cmpeq_epi8(data_vec, zero_vec);
            int mask = _mm256_movemask_epi8(cmp);

            if ((uint32_t)mask != 0xFFFFFFFF) {
                return idx + __builtin_ctz(~mask);
            }
            idx += simd_width;
        }
#elif defined(USE_NEON)
        const size_t simd_width = 16;
        uint8x16_t zero_vec = vdupq_n_u8(0);

        while (idx + simd_width <= size) {
            uint8x16_t data_vec = vld1q_u8(values + idx);
            uint8x16_t cmp = vceqq_u8(data_vec, zero_vec);

            if (vminvq_u8(cmp) == 0) {
                for (size_t i = 0; i < simd_width; i++) {
                    if (values[idx + i] != 0) {
                        return idx + i;
                    }
                }
            }
            idx += simd_width;
        }
#elif defined(USE_SVE2)
        svbool_t pg = svwhilelt_b8(idx, size);
        while (svptest_any(svptrue_b8(), pg)) {
            svuint8_t data_vec = svld1_u8(pg, values + idx);
            svbool_t cmp = svcmpne_n_u8(pg, data_vec, 0);

            if (svptest_any(pg, cmp)) {
                svbool_t before_match = svbrkb_z(pg, cmp);
                return idx + svcntp_b8(pg, before_match);
            }
            idx += svcntb();
            pg = svwhilelt_b8(idx, size);
        }
        return idx;
#endif

        // Scalar fallback
        while (idx < size) {
            if (values[idx] != 0) return idx;
            idx++;
        }
        return idx;
    }

    // =========================================================================
    // MDBX storage methods
    // =========================================================================

    InvertedIndex::PostingListView
    InvertedIndex::getReadOnlyPostingList(MDBX_txn* txn, uint32_t term_id) const
    {
        MDBX_val key;
        uint32_t tid = term_id;
        key.iov_base = &tid;
        key.iov_len = sizeof(uint32_t);

        MDBX_val data;
        int rc = mdbx_get(txn, term_postings_dbi_, &key, &data);

        if (rc == MDBX_SUCCESS && data.iov_len >= sizeof(PostingListHeader)) {
            const PostingListHeader* header = (const PostingListHeader*)data.iov_base;

            if (header->version != 5) {
                LOG_ERROR("Unsupported posting list version: " << (int)header->version);
                return {nullptr, nullptr, 0, 0, 0.0f};
            }

            uint32_t n = header->n;
            const uint8_t* ptr =
                (const uint8_t*)data.iov_base + sizeof(PostingListHeader);
            const uint32_t* doc_ids = (const uint32_t*)ptr;

#if defined(NDD_INV_IDX_STORE_FLOATS)
            uint8_t vbits = 32;
            const void* values = ptr + n * sizeof(uint32_t);
            size_t required = sizeof(PostingListHeader)
                              + n * sizeof(uint32_t) + n * sizeof(float);
#else
            uint8_t vbits = 8;
            const void* values = ptr + n * sizeof(uint32_t);
            size_t required = sizeof(PostingListHeader)
                              + n * sizeof(uint32_t) + n * sizeof(uint8_t);
#endif

            if (data.iov_len < required) {
                return {nullptr, nullptr, 0, 0, 0.0f};
            }

            return {doc_ids, values, n, vbits, header->max_value};
        }

        return {nullptr, nullptr, 0, 0, 0.0f};
    }

    std::vector<PostingListEntry>
    InvertedIndex::loadPostingList(MDBX_txn* txn, uint32_t term_id,
                                    uint32_t* out_live_count,
                                    float* out_max_value) const
    {
        MDBX_val key;
        uint32_t tid = term_id;
        key.iov_base = &tid;
        key.iov_len = sizeof(uint32_t);

        MDBX_val data;
        int rc = mdbx_get(txn, term_postings_dbi_, &key, &data);

        std::vector<PostingListEntry> entries;
        if (rc != MDBX_SUCCESS || data.iov_len < sizeof(PostingListHeader)) {
            return entries;
        }

        const PostingListHeader* header = (const PostingListHeader*)data.iov_base;
        if (header->version != 5) {
            LOG_ERROR("Unsupported posting list version: " << (int)header->version);
            return entries;
        }

        uint32_t n = header->n;

        if (out_live_count){
            *out_live_count = header->live_count;
        }

        if (out_max_value){
            *out_max_value = header->max_value;
        }

        entries.resize(n);

        const uint8_t* ptr =
            (const uint8_t*)data.iov_base + sizeof(PostingListHeader);
        const uint32_t* doc_ids = (const uint32_t*)ptr;

#if defined(NDD_INV_IDX_STORE_FLOATS)
        const float* vals = (const float*)(ptr + n * sizeof(uint32_t));
        for (uint32_t i = 0; i < n; i++) {
            entries[i].doc_id = doc_ids[i];
            entries[i].value = vals[i];
        }
#else
        const uint8_t* vals = ptr + n * sizeof(uint32_t);
        for (uint32_t i = 0; i < n; i++) {
            entries[i].doc_id = doc_ids[i];
            entries[i].value = dequantize(vals[i], header->max_value);
        }
#endif

        return entries;
    }

    bool InvertedIndex::savePostingList(MDBX_txn* txn,
                                        uint32_t term_id,
                                        const std::vector<PostingListEntry>& entries,
                                        uint32_t live_count,
                                        float max_val)
    {
        MDBX_val key;
        uint32_t tid = term_id;
        key.iov_base = &tid;
        key.iov_len = sizeof(uint32_t);

        uint32_t n = (uint32_t)entries.size();

        PostingListHeader header;
        header.version = 5;
        header.n = n;
        header.live_count = live_count;
        header.max_value = max_val;

#if defined(NDD_INV_IDX_STORE_FLOATS)
        size_t value_size = sizeof(float);
#else
        size_t value_size = sizeof(uint8_t);
#endif

        size_t total_size = sizeof(PostingListHeader)
                            + n * sizeof(uint32_t) + n * value_size;

        std::vector<uint8_t> buffer(total_size);
        std::memcpy(buffer.data(), &header, sizeof(PostingListHeader));

        uint8_t* ptr = buffer.data() + sizeof(PostingListHeader);

        uint32_t* doc_ids_out = (uint32_t*)ptr;
        for (uint32_t i = 0; i < n; i++) {
            doc_ids_out[i] = entries[i].doc_id;
        }
        ptr += n * sizeof(uint32_t);

#if defined(NDD_INV_IDX_STORE_FLOATS)
        float* vals_out = (float*)ptr;
        for (uint32_t i = 0; i < n; i++) {
            vals_out[i] = entries[i].value;
        }
#else
        uint8_t* vals_out = ptr;
        for (uint32_t i = 0; i < n; i++) {
            vals_out[i] = quantize(entries[i].value, max_val);
        }
#endif

        MDBX_val mdata;
        mdata.iov_base = buffer.data();
        mdata.iov_len = buffer.size();

        int rc = mdbx_put(txn, term_postings_dbi_, &key, &mdata, MDBX_UPSERT);
        if (rc != 0) {
            LOG_ERROR("Failed to save posting list for term "
                      << term_id << ": " << mdbx_strerror(rc));
            return false;
        }

        term_info_[term_id] = max_val;
        return true;
    }

    bool InvertedIndex::deletePostingList(MDBX_txn* txn, uint32_t term_id) {
        MDBX_val key;
        uint32_t tid = term_id;
        key.iov_base = &tid;
        key.iov_len = sizeof(uint32_t);

        int rc = mdbx_del(txn, term_postings_dbi_, &key, nullptr);
        if (rc == MDBX_SUCCESS || rc == MDBX_NOTFOUND) {
            term_info_.erase(term_id);
            return true;
        }
        return false;
    }

    // =========================================================================
    // Startup: populate term_info_ from stored posting list headers
    // =========================================================================

    bool InvertedIndex::loadTermInfo() {
        MDBX_txn* txn;
        int rc = mdbx_txn_begin(env_, nullptr, MDBX_TXN_RDONLY, &txn);
        if (rc != 0) {
            LOG_ERROR("Failed to begin transaction for loading term info: "
                        << mdbx_strerror(rc));
            return false;
        }

        MDBX_cursor* cursor;
        rc = mdbx_cursor_open(txn, term_postings_dbi_, &cursor);
        if (rc != 0) {
            mdbx_txn_abort(txn);
            return false;
        }

        MDBX_val key, data;
        rc = mdbx_cursor_get(cursor, &key, &data, MDBX_FIRST);
        while (rc == MDBX_SUCCESS) {
            if (key.iov_len == sizeof(uint32_t)
                && data.iov_len >= sizeof(PostingListHeader))
            {
                uint32_t term_id;
                std::memcpy(&term_id, key.iov_base, sizeof(uint32_t));

                const PostingListHeader* header =
                    (const PostingListHeader*)data.iov_base;
                term_info_[term_id] = header->max_value;
            }
            rc = mdbx_cursor_get(cursor, &key, &data, MDBX_NEXT);
        }

        mdbx_cursor_close(cursor);
        mdbx_txn_abort(txn);
        return true;
    }

    // =========================================================================
    // Add / remove internals
    // =========================================================================

    bool InvertedIndex::addDocumentsBatchInternal(
        MDBX_txn* txn,
        const std::vector<std::pair<ndd::idInt, SparseVector>>& docs)
    {
        std::unordered_map<uint32_t, std::vector<std::pair<ndd::idInt, float>>> term_updates;

        for (size_t d = 0; d < docs.size(); d++) {
            ndd::idInt doc_id = docs[d].first;
            const SparseVector& vec = docs[d].second;
            for (size_t i = 0; i < vec.indices.size(); i++) {
                term_updates[vec.indices[i]].push_back(
                    std::make_pair(doc_id, vec.values[i]));
            }
        }

        for (auto it = term_updates.begin(); it != term_updates.end(); ++it) {
            uint32_t term_id = it->first;
            std::vector<std::pair<ndd::idInt, float>>& updates = it->second;

            std::sort(updates.begin(), updates.end());

            std::vector<PostingListEntry> existing = loadPostingList(txn, term_id);

            std::vector<PostingListEntry> merged;
            merged.reserve(existing.size() + updates.size());

            uint32_t live_count = 0;
            float max_val = 0.0f;

            /* updates the live_count and max_val */
            auto track = [&](const PostingListEntry& e) {
                merged.push_back(e);
                if (e.value > 0.0f) {
                    live_count++;
                    if (e.value > max_val) max_val = e.value;
                }
            };

            size_t ei = 0;
            size_t ui = 0;
            while (ei < existing.size() && ui < updates.size()) {
                ndd::idInt existing_id = existing[ei].doc_id;
                ndd::idInt update_id = updates[ui].first;
                if (existing_id < update_id) {
                    track(existing[ei]);
                    ei++;
                } else if (existing_id > update_id) {
                    track(PostingListEntry(update_id, updates[ui].second));
                    ui++;
                } else {
                    track(PostingListEntry(update_id, updates[ui].second));
                    ei++;
                    ui++;
                }
            }
            while (ei < existing.size()) {
                track(existing[ei]);
                ei++;
            }
            while (ui < updates.size()) {
                track(PostingListEntry(updates[ui].first, updates[ui].second));
                ui++;
            }

            if (!savePostingList(txn, term_id, merged, live_count, max_val)) {
                LOG_ERROR("Failed to save posting list for term " << term_id);
                return false;
            }
        }

        return true;
    }

    bool InvertedIndex::removeDocumentInternal(MDBX_txn* txn,
                                          ndd::idInt doc_id,
                                          const SparseVector& vec)
    {
        for (size_t i = 0; i < vec.indices.size(); i++) {
            uint32_t term_id = vec.indices[i];

            uint32_t live_count = 0;
            float max_val = 0.0f;
            std::vector<PostingListEntry> entries =
                loadPostingList(txn, term_id, &live_count, &max_val);
            if (entries.empty()) continue;

            // Binary search for the doc_id in the sorted list.
            size_t lo = 0;
            size_t hi = entries.size();
            while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (entries[mid].doc_id < doc_id) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }

            if (lo < entries.size() && entries[lo].doc_id == doc_id) {
                // Only decrement live_count if the entry is actually live.
                uint32_t new_live = live_count;
                if(entries[lo].value > 0.0f){
                    //this doc was live earlier.
                    new_live -= 1;
                }

                entries[lo].value = 0.0f; // create a tombstone

                // Compact when tombstone ratio exceeds the configured threshold.
                uint32_t total = (uint32_t)entries.size();
                float ratio = (float)(total - new_live) / (float)total;

                if (ratio >= settings::INV_IDX_COMPACTION_TOMBSTONE_RATIO) {
                    //begin compaction

                    float compacted_max = 0.0f;
                    size_t write = 0;
                    for (size_t j = 0; j < entries.size(); j++) {
                        if (entries[j].value > 0.0f) {
                            if (entries[j].value > compacted_max)
                                compacted_max = entries[j].value;

                            write += 1;
                            entries[write] = entries[j];
                        }
                    }
                    entries.resize(write);
                    new_live = (uint32_t)write;
                    max_val = compacted_max;

                    if (entries.empty()) {
                        deletePostingList(txn, term_id);
                        continue;
                    }
                }

                if (!savePostingList(txn, term_id, entries, new_live, max_val)) {
                    return false;
                }
            }
        }

        return true;
    }

    // =========================================================================
    // Pruning
    // =========================================================================

    void InvertedIndex::pruneLongest(std::vector<PostingListIterator*>& iters,
                                float min_score)
    {
        if (iters.size() < 2) return;

        size_t longest_idx = 0;
        uint32_t longest_rem = 0;
        for (size_t i = 0; i < iters.size(); i++) {
            uint32_t rem = iters[i]->remainingEntries();
            if (rem > longest_rem) {
                longest_rem = rem;
                longest_idx = i;
            }
        }

        if (longest_idx != 0) {
            PostingListIterator* tmp = iters[0];
            iters[0] = iters[longest_idx];
            iters[longest_idx] = tmp;
        }

        PostingListIterator* longest = iters[0];
        if (longest->current_doc_id == EXHAUSTED_DOC_ID) return;

        ndd::idInt longest_doc = longest->current_doc_id;

        // Find the earliest doc_id in all OTHER lists.
        ndd::idInt others_min = EXHAUSTED_DOC_ID;
        for (size_t i = 1; i < iters.size(); i++) {
            if (iters[i]->current_doc_id < others_min) {
                others_min = iters[i]->current_doc_id;
            }
        }

        // If another list has docs at or before ours, we can't prune:
        // those docs might appear in multiple lists and need full scoring.
        if (others_min <= longest_doc) return;

        float max_possible = longest->upperBound();

        if (max_possible <= min_score) {
            if (others_min == EXHAUSTED_DOC_ID) {
                // All other lists are done - skip everything remaining.
                longest->current_doc_id = EXHAUSTED_DOC_ID;
                longest->current_entry_idx = longest->data_size;
            } else {
                // Jump to where other lists resume.
                longest->advance(others_min);
            }
        }
    }

    // =========================================================================
    // PostingListIterator methods
    // =========================================================================

    void InvertedIndex::PostingListIterator::init(const PostingListView& view) {
        doc_ids = view.doc_ids;
        values_ptr = view.values;
        data_size = view.count;
        value_bits = view.value_bits;
        max_value = view.max_value;
        current_entry_idx = 0;

        if (data_size == 0 || doc_ids == nullptr) {
            current_doc_id = EXHAUSTED_DOC_ID;
            return;
        }

        advanceToNextLive();
    }

    void InvertedIndex::PostingListIterator::advanceToNextLive() {
        if (value_bits == 32) {
            const float* vals = (const float*)values_ptr;
            while (current_entry_idx < data_size && vals[current_entry_idx] <= 0.0f) {
                current_entry_idx++;
            }
        } else {
            current_entry_idx = (uint32_t)index->findNextLiveSIMD(
                (const uint8_t*)values_ptr,
                data_size,
                current_entry_idx);
        }

        if (current_entry_idx >= data_size) {
            current_doc_id = EXHAUSTED_DOC_ID;
        } else {
            current_doc_id = doc_ids[current_entry_idx];
        }
    }

    void InvertedIndex::PostingListIterator::next() {
        current_entry_idx++;
        advanceToNextLive();
    }

    void InvertedIndex::PostingListIterator::advance(ndd::idInt target_doc_id) {
        if (current_doc_id >= target_doc_id) {
            return;
        }

        current_entry_idx = (uint32_t)index->findDocIdSIMD(
            doc_ids, data_size, current_entry_idx, target_doc_id);

        advanceToNextLive();
    }


}  // namespace ndd