#include "tool_common.hpp"
#include <iostream>
#include <sstream>
int main(int argc,char **argv) {
    using namespace amio::tools;
    Options options; std::string error;
    if(!parse_options(argc,argv,options,error)){ CliResult r{2,"",error};r.errors.push_back("commands: capabilities, inspect, validate, copy, diagnostics"); return ResultWriter(options.json,std::cout,std::cerr).write(r); }
    CliResult result;
    if(options.command=="capabilities") result=run_capabilities(options);
    else if(options.command=="inspect") result=run_inspect(options);
    else if(options.command=="validate") result=run_validate(options);
    else if(options.command=="copy") result=run_copy(options);
    else if(options.command=="diagnostics") result=run_diagnostics(options);
    else { result.exit_code=2;result.command=options.command;result.summary="unknown command";result.errors.push_back("commands: capabilities, inspect, validate, copy, diagnostics"); }
    const auto exit_code=ResultWriter(options.json,std::cout,std::cerr).write(result);
    finalize_cli_mpi();
    return exit_code;
}
