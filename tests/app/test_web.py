import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import chatbot.web as web
from chatbot.chat_service import ChatEvent
from chatbot.web import build_service, create_app, format_sse


ROOT = Path(__file__).resolve().parents[2]


def _run_node(script: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.fail("node is required for browser behavior tests")
    result = subprocess.run(
        [node, "-e", script], cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_format_sse_encodes_event_and_json_data():
    assert format_sse(ChatEvent("token", {"content": "你好"})) == (
        'event: token\ndata: {"content": "你好"}\n\n'
    )


def _configure_legacy_service(monkeypatch, records, service_factory=None):
    class FakeLlm:
        pass
    monkeypatch.setattr("chatbot.web.load_config", lambda argv: object())
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr("chatbot.web.load_profile", lambda: {})
    monkeypatch.setattr("chatbot.web.format_profile", lambda profile: "")
    monkeypatch.setattr("chatbot.web.build_runtime_llms", lambda config: (FakeLlm(), FakeLlm()))
    monkeypatch.setattr("chatbot.web.build_chain", lambda llm, profile_text: object())
    monkeypatch.setattr("chatbot.web.load_memory_config", lambda: type("MemoryConfig", (), {"enabled": False, "db_path": "ignored", "max_results": 5})())
    monkeypatch.setattr("chatbot.web.build_memory_provider", lambda config: object())
    if service_factory is not None:
        monkeypatch.setattr("chatbot.web.ChatService", service_factory)


def test_build_service_does_not_duplicate_session_history(monkeypatch):
    from chatbot.core.llm import get_session_history, store
    records = [{"role": "human", "content": "hello"}, {"role": "ai", "content": "hi"}]
    _configure_legacy_service(monkeypatch, records)
    store.clear()
    build_service(); build_service()
    assert [message.content for message in get_session_history("default").messages] == ["hello", "hi"]


def test_build_service_passes_memory_provider(monkeypatch):
    captured = {}; chat_llm = object()
    class FakeConfig:
        emotion_interval = 5; emotion_llm = object()
    class FakeService:
        def __init__(self, chain, config, emotion_llm, **kwargs): captured.update(kwargs)
    monkeypatch.setattr("chatbot.web.load_config", lambda argv: FakeConfig())
    monkeypatch.setattr("chatbot.web.load_history", lambda: [])
    monkeypatch.setattr("chatbot.web.load_profile", lambda: {})
    monkeypatch.setattr("chatbot.web.format_profile", lambda profile: "")
    monkeypatch.setattr("chatbot.web.build_runtime_llms", lambda config: (chat_llm, object()))
    monkeypatch.setattr("chatbot.web.init_session_history", lambda session_id, records: None)
    monkeypatch.setattr("chatbot.web.build_chain", lambda llm, profile_text: object())
    monkeypatch.setattr("chatbot.web._latest_emotion_for_records", lambda records: None)
    monkeypatch.setattr("chatbot.web.load_memory_config", lambda: type("MemoryConfig", (), {"enabled": False, "db_path": "ignored", "max_results": 3})())
    monkeypatch.setattr("chatbot.web.build_memory_provider", lambda config: "memory-provider")
    monkeypatch.setattr("chatbot.web.ChatService", FakeService)
    service = build_service()
    assert service.chat_llm is chat_llm
    assert captured["memory_provider"] == "memory-provider"
    assert captured["memory_max_results"] == 3


def test_build_service_uses_latest_successful_emotion(monkeypatch):
    records = [{"role": "human", "content": f"q{i}"} for i in range(5)]; captured = {}
    def fake_service(chain, config, emotion_llm, initial_records=None, initial_emotion="", **kwargs):
        captured.update(records=initial_records, emotion=initial_emotion); return SimpleNamespace()
    _configure_legacy_service(monkeypatch, records, fake_service)
    monkeypatch.setattr("chatbot.web.load_analysis_records", lambda: [{"timestamp": "t", "turn_count": 5, "emotion_interval": 5, "input": "Dialogue context: q0</s>q1</s>q2</s>q3</s>q4", "emotion": "sad", "success": True}])
    build_service()
    assert captured == {"records": records, "emotion": "sad"}


def test_build_service_restores_latest_structured_emotion_state(monkeypatch):
    records = [{"role": "human", "content": f"q{i}"} for i in range(5)]; captured = {}
    def fake_service(chain, config, emotion_llm, initial_emotion="", initial_emotion_state=None, **kwargs):
        captured.update(emotion=initial_emotion, state=initial_emotion_state); return SimpleNamespace()
    _configure_legacy_service(monkeypatch, records, fake_service)
    monkeypatch.setattr("chatbot.web.load_analysis_records", lambda: [{"timestamp": "t", "turn_count": 5, "emotion_interval": 5, "input": "Dialogue context: q0</s>q1</s>q2</s>q3</s>q4", "emotion": "anxious", "success": True, "state": {"primary_emotion": "anxious", "confidence": 0.83, "secondary_emotions": [], "evidence": "e", "reply_strategy": "calm", "trajectory_note": "x", "safety_level": "normal"}}])
    build_service()
    assert captured["emotion"] == "anxious"
    assert captured["state"].primary_emotion == "anxious"
    assert captured["state"].confidence == 0.83


def test_build_service_ignores_emotion_when_history_is_too_short(monkeypatch):
    records = [{"role": "human", "content": "hello"}]; captured = {}
    def fake_service(chain, config, emotion_llm, initial_emotion="", **kwargs):
        captured["emotion"] = initial_emotion; return SimpleNamespace()
    _configure_legacy_service(monkeypatch, records, fake_service)
    monkeypatch.setattr("chatbot.web.load_analysis_records", lambda: [{"turn_count": 5, "emotion": "sad", "success": True}])
    build_service()
    assert captured["emotion"] == ""


def test_profile_onboarding_questions_endpoint_and_legacy_payload_adapter():
    client = TestClient(create_app())
    assert client.get("/api/profile/onboarding/questions").json() == {"questions": web.ONBOARDING_QUESTIONS}
    class LegacyRequest:
        def dict(self): return {"feedback": "accurate", "message_id": "ai_1"}
    assert web._request_payload(LegacyRequest()) == {"feedback": "accurate", "message_id": "ai_1"}


def test_static_assets_and_accessible_thread_controls_exist():
    index = (ROOT / "chatbot/static/index.html").read_text(encoding="utf-8")
    assert 'id="thread-list"' in index
    assert 'id="new-thread-button"' in index
    assert 'id="delete-thread-button"' in index
    assert 'aria-label="对话列表"' in index
    assert 'role="dialog"' in index and 'aria-modal="true"' in index


NODE_DOM = r'''
class Element {
  constructor(name) {
    this.name=name; this.children=[]; this.parent=null; this.listeners={}; this.attributes={}; this.elements={};
    this.textContent=""; this.className=""; this.value=""; this.disabled=false; this.hidden=false;
    this.scrollTop=0; this.scrollHeight=0;
  }
  appendChild(c){c.parent=this;this.children.push(c);if(this.name==="form"){const visit=n=>{if(n.name&&n.name!=="textarea"&&n.name!=="button")this.elements[n.name]=n;(n.children||[]).forEach(visit);};visit(c);}return c;}
  insertBefore(c,n){c.parent=this;const i=this.children.indexOf(n);i<0?this.children.push(c):this.children.splice(i,0,c);return c;}
  replaceChildren(...items){this.children=[];items.forEach(c=>this.appendChild(c));}
  addEventListener(n,f){this.listeners[n]=f;} setAttribute(n,v){this.attributes[n]=v;}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(c=>c!==this);}
  focus(){} requestSubmit(){} set innerHTML(v){this.children=[];this._html=v;} get innerHTML(){return this._html||"";}
}
const ids=["messages","chat-form","message-input","send-button","emotion-status","safety-status","emotion-timeline","thread-list","new-thread-button","delete-thread-button"];
const elements=Object.fromEntries(ids.map(id=>[`#${id}`,new Element(id)]));
function response(body,status=200){return {ok:status<400,status,json:async()=>body};}
'''


def test_static_app_bootstrap_reuses_identity_and_renders_thread_controls():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");
const store=new Map([["hdu_erc_client_id","c_signed"],["hdu_erc_thread_id","thread-1"]]);const calls=[];
const fetch=async(url,options={})=>{calls.push({url,options});
 if(url==="/api/clients/c_signed/threads")return response({threads:[{thread_id:"thread-1",title:"一"},{thread_id:"thread-2",title:"二"}]});
 if(url==="/api/clients/c_signed/threads/thread-1")return response({messages:[],emotion:null});
 if(url==="/api/clients/c_signed/threads/thread-1/emotion-timeline?limit=5")return response({timeline:[]});
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"00000000-0000-4000-8000-000000000001"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(()=>{try{assert.equal(store.get("hdu_erc_client_id"),"c_signed");assert.equal(store.get("hdu_erc_thread_id"),"thread-1");assert.equal(elements["#thread-list"].children.length,2);assert.equal(calls[0].url,"/api/clients/c_signed/threads");assert.ok(!calls.some(c=>c.url==="/api/clients/bootstrap"));}catch(e){console.error(e);process.exit(1);}}));
''')


def test_static_app_invalid_client_rebootstraps_and_thread_actions_are_server_backed():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");
const store=new Map([["hdu_erc_client_id","bad"],["hdu_erc_thread_id","old"]]);const calls=[];
const fetch=async(url,options={})=>{calls.push({url,options});
 if(url==="/api/clients/bad/threads")return response({},401);
 if(url==="/api/clients/bootstrap")return response({client_id:"c_new",thread:{thread_id:"t1",title:"一"}},201);
 if(url==="/api/clients/c_new/threads/t1")return response({messages:[],emotion:null});
 if(url==="/api/clients/c_new/threads/t1/emotion-timeline?limit=5")return response({timeline:[]});
 if(url==="/api/clients/c_new/threads"&&options.method==="POST")return response({thread:{thread_id:"t2",title:"二"}},201);
 if(url==="/api/clients/c_new/threads/t2"&&options.method==="DELETE")return response({},204);
 if(url==="/api/clients/c_new/threads/t2")return response({messages:[],emotion:null});
 if(url==="/api/clients/c_new/threads/t2/emotion-timeline?limit=5")return response({timeline:[]});
 if(url==="/api/clients/c_new/threads")return response({threads:[{thread_id:"t1",title:"一"}]});
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"00000000-0000-4000-8000-000000000001"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(async()=>{try{assert.equal(store.get("hdu_erc_client_id"),"c_new");assert.equal(store.get("hdu_erc_thread_id"),"t1");await context.__HDU_ERC_TEST__.createThread();assert.equal(store.get("hdu_erc_thread_id"),"t2");await context.__HDU_ERC_TEST__.deleteCurrentThread();assert.equal(store.get("hdu_erc_thread_id"),"t1");assert.ok(calls.some(c=>c.url==="/api/clients/c_new/threads/t2"&&c.options.method==="DELETE"));}catch(e){console.error(e);process.exit(1);}}));
''')


def test_static_401_recovery_retries_reads_but_not_feedback_mutation():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");
const store=new Map([["hdu_erc_client_id","old"],["hdu_erc_thread_id","old-thread"]]);const calls=[];let oldSnapshots=0;
const fetch=async(url,options={})=>{calls.push({url,options});
 if(url==="/api/clients/old/threads")return response({threads:[{thread_id:"old-thread",title:"旧"}]});
 if(url==="/api/clients/old/threads/old-thread"){oldSnapshots++;return response({},401);}
 if(url==="/api/clients/bootstrap")return response({client_id:"new",thread:{thread_id:"new-thread",title:"新"}},201);
 if(url==="/api/clients/new/threads/new-thread")return response({messages:[{role:"ai",content:"新身份内容",id:"a1"}],emotion:null});
 if(url==="/api/clients/new/threads/new-thread/emotion-timeline?limit=5")return response({timeline:[]});
 if(url==="/api/clients/new/threads/new-thread/messages/a1/feedback")return response({},401);
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"id"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(async()=>{try{assert.equal(store.get("hdu_erc_client_id"),"new");assert.equal(store.get("hdu_erc_thread_id"),"new-thread");assert.equal(elements["#messages"].children[0].children[0].textContent,"新身份内容");const controls=elements["#messages"].children[0].children[1];await controls.children[0].listeners.click();assert.equal(calls.filter(c=>c.url.includes("/feedback")).length,1);assert.notEqual(store.get("hdu_erc_client_id"),"old");assert.ok(elements["#emotion-status"].textContent.includes("身份已更新")||controls.children.at(-1).textContent.includes("失败"));}catch(e){console.error(e);process.exit(1);}}));
''')


def test_static_app_first_launch_bootstraps_and_deleting_last_thread_creates_one():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");
const store=new Map();const calls=[];
const fetch=async(url,options={})=>{calls.push({url,options});
 if(url==="/api/clients/bootstrap")return response({client_id:"c_first",thread:{thread_id:"t1",title:"一"}},201);
 if(url==="/api/clients/c_first/threads/t1"&&options.method==="DELETE")return response({},204);
 if(url==="/api/clients/c_first/threads"){
   if(options.method==="POST")return response({thread:{thread_id:"t2",title:"二"}},201);
   return response({threads:[]});
 }
 if(url==="/api/clients/c_first/threads/t1"||url==="/api/clients/c_first/threads/t2")return response({messages:[],emotion:null});
 if(url.endsWith("/emotion-timeline?limit=5"))return response({timeline:[]});
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"id"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(async()=>{try{assert.equal(store.get("hdu_erc_client_id"),"c_first");assert.equal(store.get("hdu_erc_thread_id"),"t1");await context.__HDU_ERC_TEST__.deleteCurrentThread();assert.equal(store.get("hdu_erc_client_id"),"c_first");assert.equal(store.get("hdu_erc_thread_id"),"t2");assert.ok(calls.some(c=>c.url==="/api/clients/c_first/threads"&&c.options.method==="POST"));}catch(e){console.error(e);process.exit(1);}}));
''')


def test_static_thread_navigation_ignores_out_of_order_snapshot_and_timeline():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");
const store=new Map([["hdu_erc_client_id","c"],["hdu_erc_thread_id","home"]]);
function deferred(){let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};}
const aSnapshot=deferred(),aTimeline=deferred();let initialized=false;
const fetch=async(url,options={})=>{
 if(url==="/api/clients/c/threads")return response({threads:[{thread_id:"home",title:"H"},{thread_id:"A",title:"A"},{thread_id:"B",title:"B"}]});
 if(url==="/api/clients/c/threads/home")return response({messages:[],emotion:null});
 if(url==="/api/clients/c/threads/home/emotion-timeline?limit=5"){initialized=true;return response({timeline:[]});}
 if(url==="/api/clients/c/threads/A")return aSnapshot.promise;
 if(url==="/api/clients/c/threads/A/emotion-timeline?limit=5")return aTimeline.promise;
 if(url==="/api/clients/c/threads/B")return response({messages:[{role:"human",content:"B内容",id:"b"}],emotion:null});
 if(url==="/api/clients/c/threads/B/emotion-timeline?limit=5")return response({timeline:[{turn_count:2,primary_emotion:"B情绪"}]});
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"id"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(async()=>{try{assert.ok(initialized);const selectingA=context.__HDU_ERC_TEST__.selectThread("A");await Promise.resolve();const selectingB=context.__HDU_ERC_TEST__.selectThread("B");await selectingB;aSnapshot.resolve(response({messages:[{role:"human",content:"A旧内容",id:"a"}],emotion:null}));aTimeline.resolve(response({timeline:[{turn_count:1,primary_emotion:"A旧情绪"}]}));await selectingA;assert.equal(store.get("hdu_erc_thread_id"),"B");assert.equal(elements["#messages"].children[0].children[0].textContent,"B内容");assert.ok(elements["#emotion-timeline"].children[0].textContent.includes("B情绪"));}catch(e){console.error(e);process.exit(1);}}));
''')


def test_static_navigation_failures_rollback_or_enter_explicit_resync_state():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");
const store=new Map([["hdu_erc_client_id","c"],["hdu_erc_thread_id","t"]]);let deleted=false;
const fetch=async(url,options={})=>{
 if(url==="/api/clients/c/threads"&&!options.method)return deleted?response({},500):response({threads:[{thread_id:"t",title:"原对话"}]});
 if(url==="/api/clients/c/threads"&&options.method==="POST")return response({},500);
 if(url==="/api/clients/c/threads/t"&&options.method==="DELETE"){deleted=true;return response({},204);}
 if(url==="/api/clients/c/threads/t")return response({messages:[],emotion:null});
 if(url.endsWith("/emotion-timeline?limit=5"))return response({timeline:[]});
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"id"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(async()=>{try{await assert.rejects(context.__HDU_ERC_TEST__.createThread());assert.equal(store.get("hdu_erc_thread_id"),"t");assert.equal(elements["#new-thread-button"].disabled,false);await assert.rejects(context.__HDU_ERC_TEST__.deleteCurrentThread());assert.equal(store.get("hdu_erc_thread_id"),undefined);assert.ok(elements["#emotion-status"].textContent.includes("重新同步")||elements["#emotion-status"].textContent.includes("刷新"));assert.equal(elements["#delete-thread-button"].disabled,false);}catch(e){console.error(e);process.exit(1);}}));
''')


def test_static_incremental_sse_parser_handles_utf8_json_crlf_multidata_and_tail():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");
const fetch=async()=>new Promise(()=>{});const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"id"},localStorage:{getItem:()=>null,setItem(){},removeItem(){}},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
(async()=>{try{const encoded=new TextEncoder().encode('event: token\r\ndata: {"content":"你好"}\r\n\r\nevent: token\ndata: {"content":"再见"}\n');const chunks=[encoded.slice(0,25),encoded.slice(25,38),encoded.slice(38,41),encoded.slice(41)];const frames=await context.__HDU_ERC_TEST__.collectSseFrames(chunks);assert.deepEqual(JSON.parse(JSON.stringify(frames)),[{event:"token",data:{content:"你好"}},{event:"token",data:{content:"再见"}}]);const multi=await context.__HDU_ERC_TEST__.collectSseFrames([new TextEncoder().encode('event: token\ndata: {"content":\ndata: "x"}\n\n')]);assert.equal(multi[0].data.content,"x");}catch(e){console.error(e);process.exit(1);}})();
''')


def test_static_snapshot_handles_complex_messages_and_preserves_regeneration_metadata():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");
const fetch=async()=>new Promise(()=>{});const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"id"},localStorage:{getItem:()=>null,setItem(){},removeItem(){}},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
try { context.__HDU_ERC_TEST__.renderSnapshot({emotion:null,messages:[
  {role:"system",content:"SYSTEM_SECRET",id:"sys"},
  {role:"tool",content:{token:"TOOL_SECRET"},id:"tool"},
  {role:"ai",content:[],id:"ai_1",regenerated:true,original_content:"旧答案",regeneration_reason:"不准确",feedback:"like",predicted_emotion:"sad"},
  {role:"ai",content:"危机回复",id:"ai_2",safety_level:"crisis",feedback:"like"},
]});
  assert.equal(elements["#messages"].children.length,4);
  assert.equal(elements["#messages"].children[0].children[0].textContent,"[系统消息]");
  assert.equal(elements["#messages"].children[1].children[0].textContent,"[工具消息]");
  const visible=elements["#messages"].children.map(item=>item.children.map(child=>child.textContent).join(" ")).join(" ");
  assert.ok(!visible.includes("SECRET"));
  const ai=elements["#messages"].children[2];
  assert.equal(ai.attributes["data-message-id"],"ai_1");
  assert.equal(ai.attributes["data-original-content"],"旧答案");
  assert.equal(ai.attributes["data-regeneration-reason"],"不准确");
  assert.equal(ai.attributes["data-predicted-emotion"],"sad");
  assert.equal(ai.children[0].textContent,"[工具调用]");
  assert.equal(elements["#safety-status"].hidden,false);
  assert.ok(elements["#safety-status"].textContent.includes("紧急"));
} catch(e) { console.error(e); process.exit(1); }
''')


def test_static_profile_feedback_and_draft_use_scoped_urls_and_accessible_controls():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");
["profile-button","profile-panel","profile-backdrop","profile-close","profile-panel-body","profile-onboarding-prompt","profile-onboarding-start","profile-onboarding-skip"].forEach(id=>elements[`#${id}`]=new Element(id));
const store=new Map([["hdu_erc_client_id","c"],["hdu_erc_thread_id","t"]]);const calls=[];let profileGets=0;
const fetch=async(url,options={})=>{calls.push({url,options});
 if(url==="/api/clients/c/threads")return response({threads:[{thread_id:"t",title:"一"}]});
 if(url==="/api/clients/c/threads/t")return response({messages:[{role:"ai",content:"回复",id:"a1",predicted_emotion:"sad",turn_count:1}],emotion:null});
 if(url==="/api/clients/c/threads/t/emotion-timeline?limit=5")return response({timeline:[{turn_count:1,primary_emotion:"sad"}]});
 if(url==="/api/clients/c/profile"&&!options.method){profileGets++;return response({profile:{preferred_name:"旧称呼"},is_empty:false});}
 if(url==="/api/clients/c/profile"&&options.method==="PUT")return response({profile:JSON.parse(options.body).profile});
 if(url==="/api/profile/onboarding/questions")return response({questions:[{key:"preferred_name",question:"怎么称呼？"}]});
 if(url==="/api/clients/c/profile/draft")return response({draft:{preferred_name:"草稿称呼"}});
 if(url.endsWith("/messages/a1/feedback"))return response({message_id:"a1",feedback:"like"});
 if(url.endsWith("/emotion-feedback"))return response({status:"saved"},201);
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"id"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{activeElement:null,querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(async()=>{try{
  const message=elements["#messages"].children[0],controls=message.children[1];
  assert.equal(controls.children[0].attributes["aria-label"],"将回复评价为有帮助");
  controls.children[3].listeners.click();const emotionChoices=controls.children[4];await emotionChoices.children[0].listeners.click();const emotionFeedback=calls.find(c=>c.url.endsWith("/emotion-feedback"));assert.equal(emotionFeedback.options.method,"POST");assert.deepEqual(JSON.parse(emotionFeedback.options.body),{message_id:"a1",feedback:"accurate",predicted_emotion:"sad",turn_count:1});assert.ok(calls.some(c=>c.url.endsWith("/emotion-timeline?limit=5")));
  await controls.children[0].listeners.click();const feedback=calls.find(c=>c.url.endsWith("/messages/a1/feedback"));assert.equal(feedback.options.method,"PATCH");assert.deepEqual(JSON.parse(feedback.options.body),{feedback:"like"});
  await elements["#profile-button"].listeners.click();await Promise.resolve();const editForm=elements["#profile-panel-body"].children[0];assert.equal(editForm.className,"profile-form");const preferred=editForm.children[0].children[0];preferred.value="新称呼";await editForm.listeners.submit({preventDefault(){}});const saved=calls.find(c=>c.options.method==="PUT");assert.deepEqual(JSON.parse(saved.options.body),{profile:{preferred_name:"新称呼"}});
  await elements["#profile-onboarding-start"].listeners.click();const questionForm=elements["#profile-panel-body"].children[0];questionForm.children[0].children[0].value="草稿称呼";await questionForm.listeners.submit({preventDefault(){}});const draft=calls.find(c=>c.url.endsWith("/profile/draft"));assert.equal(draft.options.method,"POST");assert.deepEqual(JSON.parse(draft.options.body),{thread_id:"t",answers:[{key:"preferred_name",answer:"草稿称呼"}]});assert.equal(elements["#profile-panel-body"].children[0].className,"profile-form");
  assert.ok(profileGets>=1);
}catch(e){console.error(e);process.exit(1);}}));
''')


def test_static_feedback_failure_disables_and_restores_all_visible_controls():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");let resolveFeedback;const pending=new Promise(resolve=>resolveFeedback=resolve);const calls=[];
const fetch=async(url,options={})=>{calls.push({url,options});if(url.endsWith("/feedback"))return pending;return new Promise(()=>{});};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"id"},localStorage:{getItem:()=>"c",setItem(){},removeItem(){}},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);context.__HDU_ERC_TEST__.setStateForTest("c","t");context.__HDU_ERC_TEST__.renderSnapshot({emotion:null,messages:[{role:"ai",content:"回复",id:"a1"}]});
(async()=>{try{const controls=elements["#messages"].children[0].children[1];controls.children[2].listeners.click();const pendingSave=controls.children[0].listeners.click();const buttons=[];const visit=n=>(n.children||[]).forEach(c=>{if(c.name==="button")buttons.push(c);visit(c);});visit(controls);assert.ok(buttons.length>=9);assert.ok(buttons.every(button=>button.disabled));resolveFeedback(response({},500));await pendingSave;assert.ok(buttons.every(button=>!button.disabled));assert.equal(controls.children.at(-1).textContent,"评价保存失败");assert.equal(calls.find(c=>c.url.endsWith("/feedback")).options.method,"PATCH");}catch(e){console.error(e);process.exit(1);}})();
''')


def test_static_turn_stream_posts_uuid_uses_done_as_success_and_reloads_server_snapshot():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");
const store=new Map([["hdu_erc_client_id","c"],["hdu_erc_thread_id","t"]]);const calls=[];let snapshots=0;
const encoder=new TextEncoder();
function streamResponse(text){const bytes=encoder.encode(text);let offset=0;return {ok:true,status:200,body:{getReader(){return {async read(){if(offset>=bytes.length)return {done:true};const value=bytes.slice(offset,offset+7);offset+=7;return {done:false,value};},releaseLock(){}};}}};}
const fetch=async(url,options={})=>{calls.push({url,options});
 if(url==="/api/clients/c/threads")return response({threads:[{thread_id:"t",title:"一"}]});
 if(url==="/api/clients/c/threads/t/emotion-timeline?limit=5")return response({timeline:[]});
 if(url==="/api/clients/c/threads/t"){snapshots++;return response({emotion:null,messages:snapshots===1?[]:[{role:"human",content:"你好",id:"h1"},{role:"ai",content:"回复",id:"a1"}]});}
 if(url==="/api/clients/c/threads/t/messages:stream")return streamResponse('event: run_started\ndata: {"request_id":"00000000-0000-4000-8000-000000000009","operation":"turn","thread_id":"t"}\n\nevent: user_message\ndata: {"message_id":"h1","role":"human","content":"你好"}\n\nevent: token\ndata: {"content":"回"}\n\nevent: token\ndata: {"content":"复"}\n\nevent: safety\ndata: {"level":"supportive","guidance":"陪伴"}\n\nevent: done\ndata: {"message_id":"a1","content":"回复"}\n\n');
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"00000000-0000-4000-8000-000000000009"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(async()=>{try{await context.__HDU_ERC_TEST__.streamMessage("你好");const request=calls.find(c=>c.url.endsWith("messages:stream"));assert.equal(request.options.method,"POST");assert.deepEqual(JSON.parse(request.options.body),{message:"你好",request_id:"00000000-0000-4000-8000-000000000009"});assert.deepEqual(elements["#messages"].children.map(x=>x.children[0].textContent),["你好","回复"]);assert.equal(elements["#safety-status"].hidden,true);assert.equal(elements["#send-button"].disabled,false);}catch(e){console.error(e);process.exit(1);}}));
''')


def test_static_superseded_stream_cannot_render_or_unlock_new_stream():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");
const store=new Map([["hdu_erc_client_id","c"],["hdu_erc_thread_id","A"]]);const encoder=new TextEncoder();
function deferred(){let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};}
const oldGate=deferred(),newGate=deferred(),oldStarted=deferred(),newStarted=deferred();let bSnapshots=0;let uuid=0;
function gatedResponse(first,gate,started){let step=0;return {ok:true,status:200,body:{getReader(){return {async read(){if(step++===0){started.resolve();return {done:false,value:encoder.encode(first)}}if(step===2){return {done:false,value:encoder.encode(await gate.promise)}}return {done:true};},releaseLock(){}};}}};}
const fetch=async(url,options={})=>{
 if(url==="/api/clients/c/threads")return response({threads:[{thread_id:"A",title:"A"},{thread_id:"B",title:"B"}]});
 if(url==="/api/clients/c/threads/A")return response({messages:[],emotion:null});
 if(url==="/api/clients/c/threads/A/emotion-timeline?limit=5")return response({timeline:[]});
 if(url==="/api/clients/c/threads/B"){bSnapshots++;return response({messages:bSnapshots===1?[]:[{role:"human",content:"new",id:"hb"},{role:"ai",content:"NEW",id:"ab"}],emotion:null});}
 if(url==="/api/clients/c/threads/B/emotion-timeline?limit=5")return response({timeline:[]});
 if(url.endsWith("/threads/A/messages:stream"))return gatedResponse('event: run_started\ndata: {"request_id":"old-id","thread_id":"A"}\n\nevent: token\ndata: {"content":"OLD1"}\n\n',oldGate,oldStarted);
 if(url.endsWith("/threads/B/messages:stream"))return gatedResponse('event: run_started\ndata: {"request_id":"new-id","thread_id":"B"}\n\nevent: token\ndata: {"content":"NEW1"}\n\n',newGate,newStarted);
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>++uuid===1?"old-id":"new-id"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(async()=>{try{const oldRun=context.__HDU_ERC_TEST__.streamMessage("old");await oldStarted.promise;await context.__HDU_ERC_TEST__.selectThread("B");const newRun=context.__HDU_ERC_TEST__.streamMessage("new");await newStarted.promise;oldGate.resolve('event: token\ndata: {"content":"OLD2"}\n\nevent: done\ndata: {"message_id":"aa","content":"OLD"}\n\n');await oldRun;assert.equal(elements["#send-button"].disabled,true);const interim=elements["#messages"].children.map(x=>x.children[0].textContent).join(" ");assert.ok(!interim.includes("OLD2"));newGate.resolve('event: token\ndata: {"content":"2"}\n\nevent: done\ndata: {"message_id":"ab","content":"NEW"}\n\n');await newRun;assert.equal(elements["#send-button"].disabled,false);assert.deepEqual(elements["#messages"].children.map(x=>x.children[0].textContent),["new","NEW"]);}catch(e){console.error(e);process.exit(1);}}));
''')


def test_static_regeneration_posts_reason_and_replaces_same_message_in_place():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");
const store=new Map([["hdu_erc_client_id","c"],["hdu_erc_thread_id","t"]]);const calls=[];let snapshots=0;const encoder=new TextEncoder();
function streamResponse(text){const bytes=encoder.encode(text);let read=false;return {ok:true,status:200,body:{getReader(){return {async read(){if(read)return {done:true};read=true;return {done:false,value:bytes};},releaseLock(){}};}}};}
const fetch=async(url,options={})=>{calls.push({url,options});
 if(url==="/api/clients/c/threads")return response({threads:[{thread_id:"t",title:"一"}]});
 if(url==="/api/clients/c/threads/t/emotion-timeline?limit=5")return response({timeline:[]});
 if(url==="/api/clients/c/threads/t"){snapshots++;return response({emotion:null,messages:[{role:"ai",content:snapshots===1?"旧":"新",id:"a1",regenerated:snapshots>1,original_content:"旧",regeneration_reason:"不准确"}]});}
 if(url==="/api/clients/c/threads/t/messages/a1/regenerate:stream")return streamResponse('event: run_started\ndata: {"request_id":"00000000-0000-4000-8000-000000000010","operation":"regenerate","thread_id":"t"}\n\nevent: token\ndata: {"content":"新"}\n\nevent: done\ndata: {"message_id":"a1","content":"新","reason":"不准确","regenerated":true}\n\n');
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"00000000-0000-4000-8000-000000000010"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(async()=>{try{const wrapper=elements["#messages"].children[0];const controls=wrapper.children[1];const status=controls.children[controls.children.length-1];await context.__HDU_ERC_TEST__.submitRegeneration(wrapper,"a1","不准确",controls,status);const request=calls.find(c=>c.url.includes("regenerate:stream"));assert.deepEqual(JSON.parse(request.options.body),{reason:"不准确",request_id:"00000000-0000-4000-8000-000000000010"});assert.equal(elements["#messages"].children.length,1);const updated=elements["#messages"].children[0];assert.equal(updated.attributes["data-message-id"],"a1");assert.equal(updated.children[0].textContent,"新");assert.equal(updated.attributes["data-original-content"],"旧");}catch(e){console.error(e);process.exit(1);}}));
''')


def test_static_app_uses_client_scoped_post_streams_and_server_only_state():
    app_js = (ROOT / "chatbot/static/app.js").read_text(encoding="utf-8")
    assert "messages:stream" in app_js and "regenerate:stream" in app_js
    assert "getReader()" in app_js and "TextDecoder" in app_js and "request_id" in app_js
    assert "new EventSource" not in app_js
    assert "/api/session" not in app_js and "/api/chat/streams" not in app_js
    assert set(re.findall(r'localStorage\.(?:getItem|setItem|removeItem)\("([^"]+)"', app_js)) <= {
        "hdu_erc_client_id", "hdu_erc_thread_id"
    }


def test_index_endpoint_and_static_js_are_served():
    app = create_app()
    client = TestClient(app)
    index, script = client.get("/"), client.get("/static/app.js")
    assert index.status_code == 200 and "text/html" in index.headers["content-type"]
    assert script.status_code == 200 and "javascript" in script.headers["content-type"]


def test_superseded_global_routes_are_removed():
    app = create_app()
    client = TestClient(app)
    requests = [("get", "/api/history", None), ("get", "/api/session", None),
                ("get", "/api/profile", None), ("put", "/api/profile", {"profile": {}}),
                ("get", "/api/emotion/timeline", None),
                ("post", "/api/messages/ai_1/feedback", {"feedback": "like"}),
                ("post", "/api/chat/streams", {"message": "old"})]
    for method, path, payload in requests:
        response = getattr(client, method)(path, json=payload) if payload is not None else getattr(client, method)(path)
        assert response.status_code == 404, path
