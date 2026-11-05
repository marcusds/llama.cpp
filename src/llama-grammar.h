#pragma once

#include "llama.h"

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <vector>

struct llama_vocab;

// grammar element type
enum llama_gretype {
    // end of rule definition
    LLAMA_GRETYPE_END            = 0,

    // start of alternate definition for rule
    LLAMA_GRETYPE_ALT            = 1,

    // non-terminal element: reference to rule
    LLAMA_GRETYPE_RULE_REF       = 2,

    // terminal element: character (code point)
    LLAMA_GRETYPE_CHAR           = 3,

    // inverse char(s) ([^a], [^a-b] [^abc])
    LLAMA_GRETYPE_CHAR_NOT       = 4,

    // modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to
    // be an inclusive range ([a-z])
    LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,

    // modifies a preceding LLAMA_GRETYPE_CHAR or
    // LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
    LLAMA_GRETYPE_CHAR_ALT       = 6,

    // any character (.)
    LLAMA_GRETYPE_CHAR_ANY       = 7,

    // terminal element: token (<[token-id]>)
    LLAMA_GRETYPE_TOKEN          = 8,

    // inverse token (!<[token-id]>)
    LLAMA_GRETYPE_TOKEN_NOT      = 9,

    // non-terminal element: permutation ({| sep : a | b? |}), value is a permutation state id
    LLAMA_GRETYPE_PERMUTE        = 10,
};

typedef struct llama_grammar_element {
    enum llama_gretype type;
    uint32_t           value; // Unicode code point, rule ID, or token ID
} llama_grammar_element;

struct llama_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct llama_grammar_candidate {
    size_t               index;
    const uint32_t     * code_points;
    llama_partial_utf8   partial_utf8;
    llama_token          id;
};

using llama_grammar_rule  = std::vector<      llama_grammar_element>;
using llama_grammar_stack = std::vector<const llama_grammar_element *>;

using llama_grammar_rules      = std::vector<llama_grammar_rule>;
using llama_grammar_stacks     = std::vector<llama_grammar_stack>;
using llama_grammar_candidates = std::vector<llama_grammar_candidate>;

#define LLAMA_GRAMMAR_PERMUTE_MAX_MEMBERS 64
#define LLAMA_GRAMMAR_PERMUTE_NO_SEP      UINT32_MAX

// members (rule ids) matched in any order, a member at most once unless repeatable, with sep between two picks
struct llama_grammar_permute {
    std::vector<uint32_t> members;
    uint64_t              required   = 0;
    uint64_t              repeatable = 0;
    uint32_t              sep        = LLAMA_GRAMMAR_PERMUTE_NO_SEP;
};

// a state is a permutation plus the mask of picked members, expanded on first use into a rule at a stable address
struct llama_grammar_permute_states {
    const std::vector<llama_grammar_permute> permutes;

    explicit llama_grammar_permute_states(std::vector<llama_grammar_permute> permutes);

    const llama_grammar_element * expand(uint32_t state);

private:
    uint32_t get_state(uint32_t permute, uint64_t mask);

    std::mutex                                        mutex;
    std::map<std::pair<uint32_t, uint64_t>, uint32_t> ids;
    std::vector<std::pair<uint32_t, uint64_t>>        keys;
    std::deque<llama_grammar_rule>                    rules;
};

// TODO: remove, needed for tests atm
const llama_grammar_rules  & llama_grammar_get_rules (const struct llama_grammar * grammar);
      llama_grammar_stacks & llama_grammar_get_stacks(      struct llama_grammar * grammar);

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `llama_grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
void llama_grammar_accept(struct llama_grammar * grammar, uint32_t chr);

std::vector<llama_grammar_candidate> llama_grammar_reject_candidates_for_stack(
        const llama_grammar_rules      & rules,
        const llama_grammar_stack      & stack,
        const llama_grammar_candidates & candidates,
        llama_grammar_permute_states   * permutes = nullptr);

struct llama_grammar_parser {
    const llama_vocab * vocab;
    std::map<std::string, uint32_t> symbol_ids;

    llama_grammar_rules rules;
    std::vector<llama_grammar_permute> permutes;

    llama_grammar_parser(const struct llama_vocab * vocab = nullptr) : vocab(vocab) {}

    llama_grammar_stack c_rules() const;

    uint32_t get_symbol_id(const char * src, size_t len);
    uint32_t generate_symbol_id(const std::string & base_name);

    void add_rule(uint32_t rule_id, const llama_grammar_rule & rule);

    const char * parse_alternates(
            const char        * src,
            const std::string & rule_name,
            uint32_t            rule_id,
            bool                is_nested,
            uint32_t            depth = 0);

    const char * parse_sequence(
            const char         * src,
            const std::string  & rule_name,
            llama_grammar_rule & rule,
            bool               is_nested,
            uint32_t           depth = 0,
            bool               is_permute_member = false);

    const char * parse_permute(
            const char        * src,
            const std::string & rule_name,
            uint32_t            depth);

    const char * parse_rule(const char * src);

    bool parse(const char * src);
    void print(FILE * file);
};

struct llama_grammar_trigger_pattern {
    std::string pattern;
    std::regex  regex;

    size_t find(const std::string & input) const;
};

struct llama_grammar {
    // maintain a list of llama_tokens and their positions in the trigger_buffer
    using token_pos = std::pair<llama_token, std::pair<size_t, size_t>>;

    // note: allow null vocab for testing (not great)
    const llama_vocab * vocab;

    const llama_grammar_rules  rules;  // TODO: shared ptr
          llama_grammar_stacks stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    llama_partial_utf8 partial_utf8;

    // lazy grammars wait for trigger words or tokens before constraining the sampling.
    // we still have trigger_tokens for non-lazy grammars to force printing of special trigger tokens.
    // (useful e.g. for tool_choice=required)
    bool                     lazy             = false;
    bool                     awaiting_trigger = false; // Initialized to true for lazy grammars only
    std::string              trigger_buffer;           // Output buffered by lazy grammar. Will be cleared once trigger is found.
    std::vector<token_pos>   trigger_buffer_positions; // Tokens buffered by lazy grammar. Used to replay when a trigger is found.
    std::vector<llama_token> trigger_tokens;           // Tokens that trigger a lazy grammar, or tokens to force printing of (even if special).
    std::vector<llama_grammar_trigger_pattern>
                             trigger_patterns;         // Regular expressions that trigger a lazy grammar. Must be a full match of the entire generated
                                                       // string, and the grammar will be given the string from the first match group onwards.

    // shared by clones: stacks point into the expanded states
    std::shared_ptr<llama_grammar_permute_states> permutes;
};

//
// internal API
//

// note: needed for tests (not great)
struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
        const llama_grammar_element ** rules,
        size_t n_rules,
        size_t start_rule_index);

struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens);

void llama_grammar_free_impl(struct llama_grammar * grammar);

struct llama_grammar * llama_grammar_clone_impl(const struct llama_grammar & grammar);

// TODO: move the API below as member functions of llama_grammar
void llama_grammar_apply_impl(
        const struct llama_grammar & grammar,
            llama_token_data_array * cur_p);

void llama_grammar_accept_impl(
              struct llama_grammar & grammar,
                       llama_token   token);

void llama_grammar_accept_str(
              struct llama_grammar & grammar,
                 const std::string & piece);

void llama_grammar_accept_token(
              struct llama_grammar & grammar,
                       llama_token   token,
                 const std::string & piece);
