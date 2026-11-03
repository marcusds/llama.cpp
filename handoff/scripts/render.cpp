// usage: render <template.jinja> <input.json>   input: {"messages":[...],"tools":[...],"add_generation_prompt":bool,"kwargs":{"thinking_mode":"chat",...}}
#include "chat.h"
#include "json.h"
#include <fstream>
#include <iostream>
#include <sstream>
static std::string slurp(const char * p) { std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str(); }
int main(int argc, char ** argv) {
    auto tmpl = slurp(argv[1]);
    auto in = common_json::parse(slurp(argv[2]));
    auto tmpls = common_chat_templates_init(nullptr, tmpl);
    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(in.at("messages"));
    if (in.contains("tools")) inputs.tools = common_chat_tools_parse_oaicompat(in.at("tools"));
    inputs.add_generation_prompt = in.value("add_generation_prompt", false);
    if (in.contains("kwargs")) for (auto && [k, v] : in.at("kwargs").items()) inputs.chat_template_kwargs[k] = v.dump();
    auto p = common_chat_templates_apply(tmpls.get(), inputs);
    std::cout << p.prompt.str() << std::flush;
    return 0;
}
