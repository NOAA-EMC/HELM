#include "tool_common.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cerrno>
#include <fcntl.h>
#include <map>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#include <variant>

namespace amio::tools {
namespace {
struct JsonValue {
    using Object=std::map<std::string,JsonValue,std::less<>>;
    using Array=std::vector<JsonValue>;
    std::variant<std::nullptr_t,bool,std::string,Object,Array> value=nullptr;
    std::string raw;
};
class JsonParser {
public:
    explicit JsonParser(const std::string &s):s_(s){}
    bool parse(JsonValue &out){ws();if(!value(out,0))return false;ws();return at_==s_.size();}
private:
    void ws(){while(at_<s_.size()&&(s_[at_]==' '||s_[at_]=='\t'||s_[at_]=='\r'||s_[at_]=='\n'))++at_;}
    bool string(std::string &decoded,std::string &raw){
        if(at_>=s_.size()||s_[at_]!='"')return false; const auto start=at_++; decoded.clear();
        while(at_<s_.size()){unsigned char c=static_cast<unsigned char>(s_[at_++]);if(c=='"'){raw=s_.substr(start,at_-start);return true;}
            if(c<0x20)return false;if(c!='\\'){decoded.push_back(static_cast<char>(c));continue;}
            if(at_>=s_.size())return false;char e=s_[at_++];switch(e){case '"':case '\\':case '/':decoded.push_back(e);break;case 'b':decoded.push_back('\b');break;case 'f':decoded.push_back('\f');break;case 'n':decoded.push_back('\n');break;case 'r':decoded.push_back('\r');break;case 't':decoded.push_back('\t');break;
                case 'u':{if(at_+4>s_.size())return false;for(int i=0;i<4;++i)if(!std::isxdigit(static_cast<unsigned char>(s_[at_+i])))return false;decoded.push_back('?');at_+=4;break;}default:return false;}}
        return false;
    }
    bool value(JsonValue &out,unsigned depth){
        if(depth>48||++nodes_>100000)return false;ws();if(at_>=s_.size())return false;const auto start=at_;char c=s_[at_];
        if(c=='{'){++at_;JsonValue::Object obj;ws();if(at_<s_.size()&&s_[at_]=='}'){++at_;out.value=std::move(obj);out.raw=s_.substr(start,at_-start);return true;}
            for(;;){ws();std::string key,key_raw;if(!string(key,key_raw))return false;ws();if(at_>=s_.size()||s_[at_++]!=':')return false;JsonValue child;if(!value(child,depth+1)||!obj.emplace(std::move(key),std::move(child)).second)return false;ws();if(at_>=s_.size())return false;char sep=s_[at_++];if(sep=='}')break;if(sep!=',')return false;}
            out.value=std::move(obj);out.raw=s_.substr(start,at_-start);return true;}
        if(c=='['){++at_;JsonValue::Array array;ws();if(at_<s_.size()&&s_[at_]==']'){++at_;out.value=std::move(array);out.raw=s_.substr(start,at_-start);return true;}
            for(;;){JsonValue child;if(!value(child,depth+1))return false;array.push_back(std::move(child));ws();if(at_>=s_.size())return false;char sep=s_[at_++];if(sep==']')break;if(sep!=',')return false;}
            out.value=std::move(array);out.raw=s_.substr(start,at_-start);return true;}
        if(c=='"'){std::string decoded;if(!string(decoded,out.raw))return false;out.value=std::move(decoded);return true;}
        if(s_.compare(at_,4,"true")==0){at_+=4;out.value=true;out.raw="true";return true;}
        if(s_.compare(at_,5,"false")==0){at_+=5;out.value=false;out.raw="false";return true;}
        if(s_.compare(at_,4,"null")==0){at_+=4;out.value=nullptr;out.raw="null";return true;}
        if(c=='-')++at_;if(at_>=s_.size())return false;if(s_[at_]=='0')++at_;else{if(s_[at_]<'1'||s_[at_]>'9')return false;while(at_<s_.size()&&std::isdigit(static_cast<unsigned char>(s_[at_])))++at_;}
        if(at_<s_.size()&&s_[at_]=='.'){++at_;auto digits=at_;while(at_<s_.size()&&std::isdigit(static_cast<unsigned char>(s_[at_])))++at_;if(at_==digits)return false;}
        if(at_<s_.size()&&(s_[at_]=='e'||s_[at_]=='E')){++at_;if(at_<s_.size()&&(s_[at_]=='+'||s_[at_]=='-'))++at_;auto digits=at_;while(at_<s_.size()&&std::isdigit(static_cast<unsigned char>(s_[at_])))++at_;if(at_==digits)return false;}
        out.raw=s_.substr(start,at_-start);out.value=out.raw;return true;
    }
    const std::string &s_;std::size_t at_=0,nodes_=0;
};
const JsonValue *member(const JsonValue &v,std::string_view key){auto o=std::get_if<JsonValue::Object>(&v.value);if(!o)return nullptr;auto it=o->find(key);return it==o->end()?nullptr:&it->second;}
const JsonValue::Array *array(const JsonValue *v){return v?std::get_if<JsonValue::Array>(&v->value):nullptr;}
bool safe_integer(const JsonValue &v){if(v.raw.empty()||v.raw.front()=='"'||v.raw=="true"||v.raw=="false"||v.raw=="null"||v.raw.find_first_of(".eE")!=std::string::npos)return false;std::int64_t n=0;auto r=std::from_chars(v.raw.data(),v.raw.data()+v.raw.size(),n);return r.ec==std::errc{}&&r.ptr==v.raw.data()+v.raw.size();}
bool safe_integer_tree(const JsonValue &v){if(safe_integer(v))return true;if(auto a=std::get_if<JsonValue::Array>(&v.value))return std::all_of(a->begin(),a->end(),safe_integer_tree);return false;}
std::string raw_member(const JsonValue &v,std::string_view key){auto p=member(v,key);return p&&safe_integer(*p)?p->raw:"null";}
std::string safe_values(const JsonValue &v,std::initializer_list<std::string_view> keys){std::string out="{";bool first=true;for(auto key:keys)if(auto p=member(v,key);p&&safe_integer_tree(*p)){if(!first)out+=',';first=false;out+=ResultWriter::quote(std::string(key))+":"+p->raw;}out+='}';return out;}
std::string safe_topology(const JsonValue &v){std::string out="{";bool first=true;auto add=[&](std::string_view key,const std::set<std::string,std::less<>> &allowed){auto p=member(v,key);if(!p)return;auto s=std::get_if<std::string>(&p->value);if(!s||!allowed.contains(*s))return;if(!first)out+=',';first=false;out+=ResultWriter::quote(std::string(key))+":"+p->raw;};add("execution_mode",{"persistent_direct","persistent_aggregated","mpi_stream"});add("progress_mode",{"caller","background"});for(auto key:{"rank","rank_count"})if(auto p=member(v,key);p&&safe_integer(*p)){if(!first)out+=',';first=false;out+=ResultWriter::quote(key)+":"+p->raw;}out+='}';return out;}
std::string safe_scope(const JsonValue &v){return safe_values(v,{"context_epoch","dataset_id","step_id","field_id","role","participant","counters","resource_current","resource_high_water","resource_ceilings"});}
bool write_exclusive(const std::filesystem::path &path,const std::string &data){
    const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);if(fd<0)return false;std::size_t at=0;bool ok=true;
    while(at<data.size()){auto n=::write(fd,data.data()+at,data.size()-at);if(n<0){if(errno==EINTR)continue;ok=false;break;}if(n==0){ok=false;break;}at+=static_cast<std::size_t>(n);}
    if(ok&&::fsync(fd)!=0)ok=false;if(::close(fd)!=0)ok=false;if(!ok)::unlink(path.c_str());return ok;
}
CliResult export_snapshot(const Options &options){
    CliResult result{0,"diagnostics","diagnostic snapshot exported"};auto it=options.values.find("--export");if(it==options.values.end()){result.exit_code=2;result.errors.push_back("--export requires a destination path");return result;}
    CliRuntime runtime;auto status=open_cli_context(options,runtime);if(status!=AMIO_OK){result.exit_code=5;result.errors.push_back("could not initialize diagnostic snapshot context: "+std::to_string(status));return result;}
    std::size_t bytes=0;status=amio_query_diagnostics(runtime.context,nullptr,0,&bytes);if(status!=AMIO_OK||bytes==0||bytes>1024*1024){result.exit_code=status==AMIO_ERR_UNSUPPORTED?3:5;result.errors.push_back("diagnostic snapshot is unavailable or exceeds 1 MiB");return result;}
    std::string json(bytes,'\0');status=amio_query_diagnostics(runtime.context,json.data(),json.size(),&bytes);if(status!=AMIO_OK){result.exit_code=5;result.errors.push_back("diagnostic snapshot query failed");return result;}json.resize(bytes-1);
    if(!write_exclusive(it->second,json)){result.exit_code=2;result.errors.push_back("snapshot destination exists or cannot be created safely");return result;}
    result.summary="wrote "+std::to_string(json.size())+" bytes to a new snapshot file";result.summary_json="{\"bytes\":"+std::to_string(json.size())+",\"created\":true}";return result;
}
}

CliResult run_diagnostics(const Options &options) {
    if(options.values.contains("--export"))return export_snapshot(options);
    CliResult result{0,"diagnostics","diagnostic snapshot summary"};
    auto it=options.values.find("--input");if(it==options.values.end()){result.exit_code=2;result.errors.push_back("usage: amio diagnostics --input SNAPSHOT [--json] or --export FILE");return result;}
    std::string data;if(!read_bounded_file(it->second,1024*1024,data)){result.exit_code=2;result.errors.push_back("snapshot must be a regular file no larger than 1 MiB");return result;}
    JsonValue root;JsonParser parser(data);if(!parser.parse(root)||!std::holds_alternative<JsonValue::Object>(root.value)){result.exit_code=2;result.errors.push_back("snapshot is invalid JSON or is not an object");return result;}
    const auto version=member(root,"api_version");if(!version||version->raw!="2"||!member(root,"topology")||!member(root,"cumulative_counters")||!member(root,"scopes")||!member(root,"events")||!member(root,"first_error")){
        result.exit_code=2;result.errors.push_back("snapshot does not match the supported diagnostics schema");return result;}
    const auto *scopes=array(member(root,"scopes"));const auto *events=array(member(root,"events"));
    result.summary="decoded "+std::to_string(scopes?scopes->size():0)+" diagnostic scope(s) and "+std::to_string(events?events->size():0)+" event(s)";
    const auto *topology=member(root,"topology");const auto *counters=member(root,"cumulative_counters");
    result.summary_json="{\"api_version\":2,\"topology\":"+safe_topology(*topology)+
        ",\"dropped_sample_count\":"+raw_member(root,"dropped_sample_count")+
        ",\"scope_count\":"+raw_member(root,"scope_count")+
        ",\"cumulative_counters\":"+safe_values(*counters,{"accepted","rejected","completed","failed","bytes_accepted","bytes_written","bytes_published","source_released_bytes","transport_received_bytes","transport_published_bytes","view_bytes","stream_skips","stream_delivered_bytes","stream_released_bytes","stream_disconnects","operator_input_bytes","operator_output_bytes"})+
        ",\"first_error_status\":"+raw_member(*member(root,"first_error"),"status")+",\"scopes\":[";
    if(scopes)for(std::size_t i=0;i<scopes->size();++i){if(i)result.summary_json+=',';const auto &scope=(*scopes)[i];result.summary_json+=safe_scope(scope);}
    result.summary_json+="],\"events\":[";
    if(events)for(std::size_t i=0;i<events->size();++i){if(i)result.summary_json+=',';result.summary_json+=safe_values((*events)[i],{"kind","context_epoch","dataset_id","step_id","field_id","role","participant"});}
    result.summary_json+="]}";
    return result;
}
} // namespace amio::tools
