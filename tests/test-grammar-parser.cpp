#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"

// TODO: shold not include libllama sources
#include "../src/llama-grammar.h"

#include <cassert>

static const char * type_str(llama_gretype type) {
    switch (type) {
        case LLAMA_GRETYPE_CHAR: return "LLAMA_GRETYPE_CHAR";
        case LLAMA_GRETYPE_CHAR_NOT: return "LLAMA_GRETYPE_CHAR_NOT";
        case LLAMA_GRETYPE_CHAR_ALT: return "LLAMA_GRETYPE_CHAR_ALT";
        case LLAMA_GRETYPE_CHAR_RNG_UPPER: return "LLAMA_GRETYPE_CHAR_RNG_UPPER";
        case LLAMA_GRETYPE_RULE_REF: return "LLAMA_GRETYPE_RULE_REF";
        case LLAMA_GRETYPE_ALT: return "LLAMA_GRETYPE_ALT";
        case LLAMA_GRETYPE_END: return "LLAMA_GRETYPE_END";
        case LLAMA_GRETYPE_PERMUTE: return "LLAMA_GRETYPE_PERMUTE";
        default: return "?";
    }
}

static void verify_parsing(const char *grammar_bytes, const std::vector<std::pair<std::string, uint32_t>> expected, const std::vector<llama_grammar_element> &expected_rules) {
    uint32_t index = 0;
    llama_grammar_parser parsed_grammar;
    parsed_grammar.parse(grammar_bytes);

    std::map<uint32_t, std::string> symbol_names;
    for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it) {
        symbol_names[it->second] = it->first;
    }

    auto print_all = [&]() {
        fprintf(stderr, "    verify_parsing(R\"\"\"(%s)\"\"\", {\n", grammar_bytes);
        for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it) {
            fprintf(stderr, "        {\"%s\", %u},\n", it->first.c_str(), it->second);
        }
        fprintf(stderr, "    }, {\n");
        for (size_t i_rule = 0; i_rule < parsed_grammar.rules.size(); i_rule++) {
            fprintf(stderr, "        // %s (index %zu)\n", symbol_names[i_rule].c_str(), i_rule);
            auto & rule = parsed_grammar.rules[i_rule];
            for (uint32_t i = 0; i < rule.size(); i++) {
                std::string rule_str;
                fprintf(stderr, "        {%s, ", type_str(rule[i].type));
                if (rule[i].type == LLAMA_GRETYPE_CHAR || rule[i].type == LLAMA_GRETYPE_CHAR_ALT ||
                    rule[i].type == LLAMA_GRETYPE_CHAR_NOT || rule[i].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
                    char c = rule[i].value;
                    if (c == '\n') {
                        fprintf(stderr, "'\\n'");
                    } else if (c == '\t') {
                        fprintf(stderr, "'\\t'");
                    } else if (c == '\r') {
                        fprintf(stderr, "'\\r'");
                    } else if (c == '\0') {
                        fprintf(stderr, "'\\0'");
                    } else {
                        fprintf(stderr, "'%c'", c);
                    }
                } else if (rule[i].type == LLAMA_GRETYPE_RULE_REF) {
                    fprintf(stderr, "/* %s */ %u", symbol_names[rule[i].value].c_str(), rule[i].value);
                } else {
                    fprintf(stderr, "%u", rule[i].value);
                }
                fprintf(stderr, "},\n");
            }
        }
        fprintf(stderr, "    });\n");
    };

    if (getenv("TEST_GRAMMAR_PARSER_PRINT_ALL")) {
        print_all();
        fprintf(stderr, "\n");
        return;
    }

    fprintf(stderr, "Testing grammar:%s\n", grammar_bytes);

    if (parsed_grammar.symbol_ids.size() != expected.size()) {
        fprintf(stderr, "Code to update expectation (set TEST_GRAMMAR_PARSER_PRINT_ALL=1 to print all):\n");
        print_all();
        assert(parsed_grammar.symbol_ids.size() == expected.size());
    }

    for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it)
    {
        std::string key = it->first;
        uint32_t value = it->second;
        std::pair<std::string, uint32_t> expected_pair = expected[index];

        // pretty print error message before asserting
        if (expected_pair.first != key || expected_pair.second != value)
        {
            fprintf(stderr, "index: %u\n", index);
            fprintf(stderr, "expected_pair: %s, %u\n", expected_pair.first.c_str(), expected_pair.second);
            fprintf(stderr, "actual_pair: %s, %u\n", key.c_str(), value);
            fprintf(stderr, "expected_pair != actual_pair\n");
            fprintf(stderr, "Code to update expectation (set TEST_GRAMMAR_PARSER_PRINT_ALL=1 to print all):\n");
            print_all();
        }

        assert(expected_pair.first == key && expected_pair.second == value);

        index++;
    }

    index = 0;
    for (auto rule : parsed_grammar.rules)
    {
        // compare rule to expected rule
        for (uint32_t i = 0; i < rule.size(); i++)
        {
            llama_grammar_element element = rule[i];
            llama_grammar_element expected_element = expected_rules[index];

            // pretty print error message before asserting
            if (expected_element.type != element.type || expected_element.value != element.value)
            {
                fprintf(stderr, "index: %u\n", index);
                fprintf(stderr, "expected_element: %s, %u\n", type_str(expected_element.type), expected_element.value);
                fprintf(stderr, "actual_element: %s, %u\n", type_str(element.type), element.value);
                fprintf(stderr, "expected_element != actual_element\n");
                fprintf(stderr, "all elements:\n");
                fprintf(stderr, "Code to update expectation (set TEST_GRAMMAR_PARSER_PRINT_ALL=1 to print all):\n");
                print_all();
            }

            assert(expected_element.type == element.type && expected_element.value == element.value);
            index++;
        }
    }
}

static void verify_failure(const char * grammar_bytes) {
    fprintf(stderr, "Testing expected failure:%s\n", grammar_bytes);
    llama_grammar_parser result;
    result.parse(grammar_bytes);
    assert(result.rules.empty() && "should have failed");
}

static std::string print_grammar(llama_grammar_parser & parsed_grammar) {
    FILE * file = tmpfile();
    parsed_grammar.print(file);
    std::string result(ftell(file), '\0');
    rewind(file);
    result.resize(fread(result.data(), 1, result.size(), file));
    fclose(file);
    return result;
}

static void verify_permute(const char * grammar_bytes, const std::vector<llama_grammar_permute> & expected, const std::string & expected_print) {
    fprintf(stderr, "Testing permutation:%s\n", grammar_bytes);
    llama_grammar_parser parsed_grammar;
    assert(parsed_grammar.parse(grammar_bytes));

    const std::string printed = print_grammar(parsed_grammar);
    bool matches = parsed_grammar.permutes.size() == expected.size() && printed == expected_print;
    for (size_t i = 0; matches && i < expected.size(); i++) {
        const auto & perm = parsed_grammar.permutes[i];
        matches = perm.members == expected[i].members && perm.required == expected[i].required &&
                  perm.repeatable == expected[i].repeatable && perm.sep == expected[i].sep;
    }
    if (!matches) {
        for (const auto & perm : parsed_grammar.permutes) {
            fprintf(stderr, "members:");
            for (const uint32_t member : perm.members) {
                fprintf(stderr, " %u", member);
            }
            fprintf(stderr, " required: %llu repeatable: %llu sep: %u\n", (unsigned long long) perm.required, (unsigned long long) perm.repeatable, perm.sep);
        }
        fprintf(stderr, "expected:\n%s\nactual:\n%s\n", expected_print.c_str(), printed.c_str());
        assert(false);
    }
}

int main()
{
    verify_failure(R"""(
        root ::= "a"{,}"
    )""");

    verify_failure(R"""(
        root ::= (((((([^x]*){0,99}){0,99}){0,99}){0,99}){0,99}){0,99}
    )""");

    verify_failure(R"""(
        root ::= "a"{,10}"
    )""");

    verify_failure(R"""(
        root ::= "a"{5000}
    )""");

    verify_failure(R"""(
        root ::= "a"{5000,}
    )""");

    verify_failure(R"""(
        root ::= "a"{5000,6000}
    )""");

    // Recursion depth limit exceeded (Issue #29495)
    {
        std::string deep_grammar = "root ::= ";
        int n_depth = 300;
        for (int i = 0; i < n_depth; ++i) {
            deep_grammar += "(";
        }
        deep_grammar += "\"x\"";
        for (int i = 0; i < n_depth; ++i) {
            deep_grammar += ")";
        }
        verify_failure(deep_grammar.c_str());
    }

    verify_parsing(R"""(
        root ::= "a"{0,5000}
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root ::= "a"{3,5000}
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"
    )""", {
        {"root", 0},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a" | [bdx-z] | [^1-3]
    )""", {
        {"root", 0},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_CHAR, 'b'},
        {LLAMA_GRETYPE_CHAR_ALT, 'd'},
        {LLAMA_GRETYPE_CHAR_ALT, 'x'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 'z'},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_CHAR_NOT, '1'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, '3'},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= a+
        a     ::= "a"
    )""", {
        {"a", 1},
        {"root", 0},
        {"root_2", 2},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* a */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
        {LLAMA_GRETYPE_END, 0},
        // a (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_END, 0},
        // root_2 (index 2)
        {LLAMA_GRETYPE_RULE_REF, /* a */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"+
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= a?
        a     ::= "a"
    )""", {
        {"a", 1},
        {"root", 0},
        {"root_2", 2},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
        {LLAMA_GRETYPE_END, 0},
        // a (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_END, 0},
        // root_2 (index 2)
        {LLAMA_GRETYPE_RULE_REF, /* a */ 1},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"?
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= a*
        a     ::= "a"
    )""", {
        {"a", 1},
        {"root", 0},
        {"root_2", 2},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
        {LLAMA_GRETYPE_END, 0},
        // a (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_END, 0},
        // root_2 (index 2)
        {LLAMA_GRETYPE_RULE_REF, /* a */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"*
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{2}
    )""", {
        {"root", 0},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{2,}
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{ 4}
    )""", {
        {"root", 0},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{2,4}
    )""", {
        {"root", 0},
        {"root_1", 1},
        {"root_2", 2},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        // root_2 (index 2)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= (expr "=" term "\n")+
        expr  ::= term ([-+*/] term)*
        term  ::= [0-9]+
    )""", {
        {"expr", 2},
        {"expr_5", 5},
        {"expr_6", 6},
        {"root", 0},
        {"root_1", 1},
        {"root_4", 4},
        {"term", 3},
        {"term_7", 7},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_4 */ 4},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_RULE_REF, /* expr */ 2},
        {LLAMA_GRETYPE_CHAR, '='},
        {LLAMA_GRETYPE_RULE_REF, /* term */ 3},
        {LLAMA_GRETYPE_CHAR, '\n'},
        {LLAMA_GRETYPE_END, 0},
        // expr (index 2)
        {LLAMA_GRETYPE_RULE_REF, /* term */ 3},
        {LLAMA_GRETYPE_RULE_REF, /* expr_6 */ 6},
        {LLAMA_GRETYPE_END, 0},
        // term (index 3)
        {LLAMA_GRETYPE_CHAR, '0'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, '9'},
        {LLAMA_GRETYPE_RULE_REF, /* term_7 */ 7},
        {LLAMA_GRETYPE_END, 0},
        // root_4 (index 4)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_4 */ 4},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        // expr_5 (index 5)
        {LLAMA_GRETYPE_CHAR, '-'},
        {LLAMA_GRETYPE_CHAR_ALT, '+'},
        {LLAMA_GRETYPE_CHAR_ALT, '*'},
        {LLAMA_GRETYPE_CHAR_ALT, '/'},
        {LLAMA_GRETYPE_RULE_REF, /* term */ 3},
        {LLAMA_GRETYPE_END, 0},
        // expr_6 (index 6)
        {LLAMA_GRETYPE_RULE_REF, /* expr_5 */ 5},
        {LLAMA_GRETYPE_RULE_REF, /* expr_6 */ 6},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        // term_7 (index 7)
        {LLAMA_GRETYPE_CHAR, '0'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, '9'},
        {LLAMA_GRETYPE_RULE_REF, /* term_7 */ 7},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= (expr "=" ws term "\n")+
        expr  ::= term ([-+*/] term)*
        term  ::= ident | num | "(" ws expr ")" ws
        ident ::= [a-z] [a-z0-9_]* ws
        num   ::= [0-9]+ ws
        ws    ::= [ \t\n]*
    )""", {
        {"expr", 2},
        {"expr_6", 6},
        {"expr_7", 7},
        {"ident", 8},
        {"ident_10", 10},
        {"num", 9},
        {"num_11", 11},
        {"root", 0},
        {"root_1", 1},
        {"root_5", 5},
        {"term", 4},
        {"ws", 3},
        {"ws_12", 12},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_5 */ 5},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_RULE_REF, /* expr */ 2},
        {LLAMA_GRETYPE_CHAR, '='},
        {LLAMA_GRETYPE_RULE_REF, /* ws */ 3},
        {LLAMA_GRETYPE_RULE_REF, /* term */ 4},
        {LLAMA_GRETYPE_CHAR, '\n'},
        {LLAMA_GRETYPE_END, 0},
        // expr (index 2)
        {LLAMA_GRETYPE_RULE_REF, /* term */ 4},
        {LLAMA_GRETYPE_RULE_REF, /* expr_7 */ 7},
        {LLAMA_GRETYPE_END, 0},
        // ws (index 3)
        {LLAMA_GRETYPE_RULE_REF, /* ws_12 */ 12},
        {LLAMA_GRETYPE_END, 0},
        // term (index 4)
        {LLAMA_GRETYPE_RULE_REF, /* ident */ 8},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_RULE_REF, /* num */ 9},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_CHAR, '('},
        {LLAMA_GRETYPE_RULE_REF, /* ws */ 3},
        {LLAMA_GRETYPE_RULE_REF, /* expr */ 2},
        {LLAMA_GRETYPE_CHAR, ')'},
        {LLAMA_GRETYPE_RULE_REF, /* ws */ 3},
        {LLAMA_GRETYPE_END, 0},
        // root_5 (index 5)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_5 */ 5},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        // expr_6 (index 6)
        {LLAMA_GRETYPE_CHAR, '-'},
        {LLAMA_GRETYPE_CHAR_ALT, '+'},
        {LLAMA_GRETYPE_CHAR_ALT, '*'},
        {LLAMA_GRETYPE_CHAR_ALT, '/'},
        {LLAMA_GRETYPE_RULE_REF, /* term */ 4},
        {LLAMA_GRETYPE_END, 0},
        // expr_7 (index 7)
        {LLAMA_GRETYPE_RULE_REF, /* expr_6 */ 6},
        {LLAMA_GRETYPE_RULE_REF, /* expr_7 */ 7},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        // ident (index 8)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 'z'},
        {LLAMA_GRETYPE_RULE_REF, /* ident_10 */ 10},
        {LLAMA_GRETYPE_RULE_REF, /* ws */ 3},
        {LLAMA_GRETYPE_END, 0},
        // num (index 9)
        {LLAMA_GRETYPE_CHAR, '0'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, '9'},
        {LLAMA_GRETYPE_RULE_REF, /* num_11 */ 11},
        {LLAMA_GRETYPE_RULE_REF, /* ws */ 3},
        {LLAMA_GRETYPE_END, 0},
        // ident_10 (index 10)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 'z'},
        {LLAMA_GRETYPE_CHAR_ALT, '0'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, '9'},
        {LLAMA_GRETYPE_CHAR_ALT, '_'},
        {LLAMA_GRETYPE_RULE_REF, /* ident_10 */ 10},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        // num_11 (index 11)
        {LLAMA_GRETYPE_CHAR, '0'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, '9'},
        {LLAMA_GRETYPE_RULE_REF, /* num_11 */ 11},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        // ws_12 (index 12)
        {LLAMA_GRETYPE_CHAR, ' '},
        {LLAMA_GRETYPE_CHAR_ALT, '\t'},
        {LLAMA_GRETYPE_CHAR_ALT, '\n'},
        {LLAMA_GRETYPE_RULE_REF, /* ws_12 */ 12},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root ::= {| "," : a | b? | c* | d+ |}
        a ::= "a"
        b ::= "b"
        c ::= "c"
        d ::= "d"
    )""", {
        {"a", 2},
        {"b", 3},
        {"c", 4},
        {"d", 5},
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_PERMUTE, 0},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_CHAR, ','},
        {LLAMA_GRETYPE_END, 0},
        // a (index 2)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_END, 0},
        // b (index 3)
        {LLAMA_GRETYPE_CHAR, 'b'},
        {LLAMA_GRETYPE_END, 0},
        // c (index 4)
        {LLAMA_GRETYPE_CHAR, 'c'},
        {LLAMA_GRETYPE_END, 0},
        // d (index 5)
        {LLAMA_GRETYPE_CHAR, 'd'},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_permute(R"""(
        root ::= "{" {| "," : a | b? | c* | d+ |} "}"
        a ::= "a"
        b ::= "b"
        c ::= "c"
        d ::= "d"
    )""", {
        {{2, 3, 4, 5}, 0b1001, 0b1100, 1},
    },
        "root ::= [{] {| root_1 : a | b? | c* | d+ |} [}] \n"
        "root_1 ::= [,] \n"
        "a ::= [a] \n"
        "b ::= [b] \n"
        "c ::= [c] \n"
        "d ::= [d] \n");

    verify_permute(R"""(
        root ::= {| ("a")? | "b"* "c" | "d" "e" |}
    )""", {
        {{1, 3, 4}, 0b110, 0, LLAMA_GRAMMAR_PERMUTE_NO_SEP},
    },
        "root ::= {| root_1? | root_3 | root_4 |} \n"
        "root_1 ::= [a] \n"
        "root_2 ::= [b] root_2 | \n"
        "root_3 ::= root_2 [c] \n"
        "root_4 ::= [d] [e] \n");

    verify_permute(R"""(
        root ::= {| "," " "? : "x" | {| ";" : "y" | "z"? |}* |} ("!" {| "a" | "b" |}){2}
    )""", {
        {{5, 6}, 0b01, 0, 4},
        {{3, 7}, 0b01, 0b10, 2},
        {{9, 10}, 0b11, 0, LLAMA_GRAMMAR_PERMUTE_NO_SEP},
    },
        "root ::= {| root_2 : root_3 | root_7* |} root_8 root_8 \n"
        "root_1 ::= [ ] | \n"
        "root_2 ::= [,] root_1 \n"
        "root_3 ::= [x] \n"
        "root_4 ::= [;] \n"
        "root_5 ::= [y] \n"
        "root_6 ::= [z] \n"
        "root_7 ::= {| root_4 : root_5 | root_6? |} \n"
        "root_8 ::= [!] {| root_9 | root_10 |} \n"
        "root_9 ::= [a] \n"
        "root_10 ::= [b] \n");

    verify_failure(R"""(
        root ::= {| |}
    )""");

    verify_failure(R"""(
        root ::= {| "," : |}
    )""");

    verify_failure(R"""(
        root ::= {| "a" | |}
    )""");

    verify_failure(R"""(
        root ::= {| "a" | "b"
    )""");

    verify_failure(R"""(
        root ::= {| "a" | "b" }
    )""");

    verify_failure(R"""(
        root ::= {| "a" | missing |}
    )""");

    verify_failure(R"""(
        root ::= {| missing : "a" | "b" |}
    )""");

    {
        std::string grammar = "root ::= {| ";
        for (int i = 0; i < 65; i++) {
            grammar += (i > 0 ? " | \"" : "\"") + std::to_string(i) + "\"";
        }
        verify_failure((grammar + " |}").c_str());
    }

    // <[1000]> = "<think>"
    // <[1001]> = "</think>"
    verify_parsing(R"""(
        root  ::= <[1000]> !<[1001]> <[1001]>
    )""", {
        {"root", 0}
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_TOKEN, 1000},
        {LLAMA_GRETYPE_TOKEN_NOT, 1001},
        {LLAMA_GRETYPE_TOKEN, 1001},
        {LLAMA_GRETYPE_END, 0},
    });

    return 0;
}
