#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <algorithm>
#include <queue>
#include <cstring>
#include <atomic>
#include <thread>
#include <shared_mutex>
#include <unordered_set>
#include <limits>
#include <cstdint>
#include <cmath>
#include "../core/types.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#    include <arm_neon.h>
#endif

#include "mdbx/mdbx.h"
#include "../utils/log.hpp"
#include "../core/types.hpp"

#include "sparse_vector.hpp"

namespace ndd {

    static constexpr ndd::idInt EXHAUSTED_DOC_ID = std::numeric_limits<ndd::idInt>::max();

    /**
     * ---------- On-disk header for a posting list ----------
     * Stored at the start of every MDBX value.
    */
#pragma pack(push, 1)
    struct PostingListHeader {
        uint8_t version = 5;       // format version
        uint32_t n = 0;            // total number of entries (including tombstones)
        uint32_t live_count = 0;   // entries with value > 0
        float max_value = 0.0f;    // largest weight in the list (used for quantization & pruning)
    };
#pragma pack(pop)

    struct PostingListEntry {
        ndd::idInt doc_id;
        float value;

        PostingListEntry() : doc_id(0), value(0.0f) {}
        PostingListEntry(ndd::idInt id, float val) : doc_id(id), value(val) {}
    };

    // ---------- Min-heap element for top-K collection ----------
    struct ScoredDoc {
        ndd::idInt doc_id;
        float score;

        ScoredDoc(ndd::idInt id, float s) : doc_id(id), score(s) {}

        bool operator<(const ScoredDoc& other) const {
            return score > other.score;  // lowest score on top
        }
    };

    class InvertedIndex {
    public:
        InvertedIndex(MDBX_env* env, size_t vocab_size);
        ~InvertedIndex() = default;

        bool initialize();

        // Insert or update a batch of documents in the index.
        bool addDocumentsBatch(MDBX_txn* txn,
                                const std::vector<std::pair<ndd::idInt, SparseVector>>& docs);

        /**
         * Remove a single document from the index by tombstoning its
         * entries (setting the weight to 0).
         */
        bool removeDocument(MDBX_txn* txn, ndd::idInt doc_id, const SparseVector& vec);

        size_t getTermCount() const;
        size_t getVocabSize() const;

        std::vector<std::pair<ndd::idInt, float>>search(const SparseVector& query,
                                                        size_t k,
                                                        const ndd::RoaringBitmap* filter = nullptr);

    private:
        MDBX_env* env_;
        // MDBX database handle: term_id -> posting list bytes
        MDBX_dbi term_postings_dbi_;
        size_t vocab_size_;

        // In-memory cache: term_id -> global max weight in that posting list.
        // Populated at startup by scanning all posting list headers.
        // Used during search for pruning upper-bound calculations.
        std::unordered_map<uint32_t, float> term_info_;

        mutable std::shared_mutex mutex_;  // readers (search) vs writers (add/remove)

        static inline uint8_t quantize(float val, float max_val);
        static inline float dequantize(uint8_t val, float max_val);

        /**
         * =====================================================================
         * PostingListView - zero-copy read into an MDBX page
         * =====================================================================
         *
         * MDBX returns a pointer directly into its memory-mapped page.
         * We never copy the data; the pointers are valid for the lifetime
         * of the read transaction.
         */
        struct PostingListView {
            const uint32_t* doc_ids;  // sorted array of document IDs
            const void* values;       // uint8_t* (quantized) or float* depending on build
            uint32_t count;           // number of entries
            uint8_t value_bits;       // 8 = quantized uint8, 32 = raw float
            float max_value;          // max weight (from header, for dequantization)
        };

        /**
         * =====================================================================
         * PostingListIterator - walks one posting list during search
         * =====================================================================
         *
         * Points into the zero-copy MDBX data. Maintains a cursor
         * (current_entry_idx / current_doc_id) that advances forward only.
         * Tombstoned entries (value == 0) are automatically skipped.
         */
        struct PostingListIterator {
            uint32_t term_id;
            float term_weight;      // query weight for this term
            float global_max;       // max stored weight in this posting list
            const InvertedIndex* index;  // back-pointer for SIMD helper access

            const uint32_t* doc_ids;  // zero-copy pointer to doc_id array
            const void* values_ptr;   // zero-copy pointer to values array
            uint32_t data_size;       // total entry count
            uint8_t value_bits;       // 8 or 32
            float max_value;          // for dequantization

            uint32_t current_entry_idx;  // cursor position in the arrays
            ndd::idInt current_doc_id;   // doc_id at cursor, or EXHAUSTED_DOC_ID

            // Set up the iterator from a PostingListView and advance to
            // the first live (non-tombstoned) entry.
            void init(const PostingListView& view);

            inline float valueAt(uint32_t idx) const {
                if (value_bits == 32) {
                    return ((const float*)values_ptr)[idx];
                }
                return dequantize(((const uint8_t*)values_ptr)[idx], max_value);
            }

            inline bool isLiveAt(uint32_t idx) const {
                if (value_bits == 32) {
                    return ((const float*)values_ptr)[idx] > 0.0f;
                }
                return ((const uint8_t*)values_ptr)[idx] > 0;
            }

            inline float currentValue() const {
                return valueAt(current_entry_idx);
            }

            // Skip tombstoned entries starting at current_entry_idx.
            // Updates current_doc_id to the next live doc, or EXHAUSTED_DOC_ID.
            void advanceToNextLive();

            // Move to the next live entry after the current one.
            void next();

            // Skip forward to the first live entry with doc_id >= target.
            // Uses SIMD binary search over the sorted doc_id array.
            void advance(ndd::idInt target_doc_id);

            // Upper bound on the score any remaining entry can contribute.
            // Used by the pruning step: if this is below the K-th best
            // score, we can skip ahead safely.
            float upperBound() const {
                return global_max * term_weight;
            }

            uint32_t remainingEntries() const {
                if (current_doc_id == EXHAUSTED_DOC_ID) return 0;
                return data_size - current_entry_idx;
            }

        private:
            // Expose dequantize to iterator inline methods
            static inline float dequantize(uint8_t val, float max_val) {
                if (max_val <= settings::NEAR_ZERO) return 0.0f;
                return (float)val * (max_val / UINT8_MAX);
            }
        };

        // =====================================================================
        // SIMD helpers
        // =====================================================================

        size_t findDocIdSIMD(const uint32_t* doc_ids,
                                size_t size,
                                size_t start_idx,
                                uint32_t target) const;

        size_t findNextLiveSIMD(const uint8_t* values,
                                size_t size,
                                size_t start_idx) const;

        // =====================================================================
        // MDBX storage methods
        // =====================================================================

        PostingListView getReadOnlyPostingList(MDBX_txn* txn, uint32_t term_id) const;

        std::vector<PostingListEntry> loadPostingList(MDBX_txn* txn, uint32_t term_id,
                                                        uint32_t* out_live_count = nullptr,
                                                        float* out_max_value = nullptr) const;

        bool savePostingList(MDBX_txn* txn,
                            uint32_t term_id,
                            const std::vector<PostingListEntry>& entries,
                            uint32_t live_count,
                            float max_val);

        bool deletePostingList(MDBX_txn* txn, uint32_t term_id);

        // =====================================================================
        // Startup
        // =====================================================================

        bool loadTermInfo();

        // =====================================================================
        // Add / remove internals
        // =====================================================================

        bool addDocumentsBatchInternal(
            MDBX_txn* txn,
            const std::vector<std::pair<ndd::idInt, SparseVector>>& docs);

        bool removeDocumentInternal(MDBX_txn* txn,
                                    ndd::idInt doc_id,
                                    const SparseVector& vec);

        // =====================================================================
        // Pruning
        // =====================================================================

        void pruneLongest(std::vector<PostingListIterator*>& iters, float min_score);
    };

}  // namespace ndd