import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import chatbot.web as web
from chatbot.models import GraphEvent
from chatbot.web import create_app, format_sse


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
    assert format_sse(GraphEvent(event="token", data={"content": "你好"})) == (
        'event: token\ndata: {"content": "你好"}\n\n'
    )


def test_profile_onboarding_questions_endpoint_and_model_payload_adapter():
    client = TestClient(create_app())
    assert client.get("/api/profile/onboarding/questions").json() == {"questions": web.ONBOARDING_QUESTIONS}
    class CompatibleRequest:
        def dict(self): return {"feedback": "accurate", "message_id": "ai_1"}
    assert web._request_payload(CompatibleRequest()) == {"feedback": "accurate", "message_id": "ai_1"}


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
  focus(){this.focused=true;} requestSubmit(){} set innerHTML(v){this.children=[];this._html=v;} get innerHTML(){return this._html||"";}
}
const ids=["messages","chat-form","message-input","send-button","emotion-status","app-status","safety-status","emotion-timeline","thread-list","new-thread-button","delete-thread-button","pending-turn","pending-turn-status","pending-retry-button"];
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


def test_static_401_recovery_retries_reads_but_not_emotion_mutation():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");
const store=new Map([["hdu_erc_client_id","old"],["hdu_erc_thread_id","old-thread"]]);const calls=[];let oldSnapshots=0;
const fetch=async(url,options={})=>{calls.push({url,options});
 if(url==="/api/clients/old/threads")return response({threads:[{thread_id:"old-thread",title:"旧"}]});
 if(url==="/api/clients/old/threads/old-thread"){oldSnapshots++;return response({},401);}
 if(url==="/api/clients/bootstrap")return response({client_id:"new",thread:{thread_id:"new-thread",title:"新"}},201);
 if(url==="/api/clients/new/threads/new-thread")return response({messages:[{role:"ai",content:"新身份内容",id:"a1"}],emotion:null});
 if(url==="/api/clients/new/threads/new-thread/emotion-timeline?limit=5")return response({timeline:[]});
 if(url==="/api/clients/new/threads/new-thread/emotion-feedback")return response({},401);
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"id"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(async()=>{try{assert.equal(store.get("hdu_erc_client_id"),"new");assert.equal(store.get("hdu_erc_thread_id"),"new-thread");assert.equal(elements["#messages"].children[0].children[0].textContent,"新身份内容");const controls=elements["#messages"].children[0].children[1];controls.children[1].listeners.click();await controls.children[2].children[0].listeners.click();assert.equal(calls.filter(c=>c.url.endsWith("/emotion-feedback")).length,1);assert.notEqual(store.get("hdu_erc_client_id"),"old");assert.ok(elements["#app-status"].textContent.includes("身份已更新")||controls.children.at(-1).textContent.includes("失败"));}catch(e){console.error(e);process.exit(1);}}));
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


def test_static_failed_thread_navigation_restores_atomic_previous_view():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");
const store=new Map([["hdu_erc_client_id","c"],["hdu_erc_thread_id","home"]]);
const fetch=async(url,options={})=>{
 if(url==="/api/clients/c/threads")return response({threads:[{thread_id:"home",title:"H"},{thread_id:"timeline-fails",title:"T"},{thread_id:"snapshot-fails",title:"S"}]});
 if(url==="/api/clients/c/threads/home")return response({messages:[{role:"ai",content:"HOME",id:"h",safety_level:"crisis"}],emotion:{primary_emotion:"calm"}});
 if(url==="/api/clients/c/threads/home/emotion-timeline?limit=5")return response({timeline:[{turn_count:1,primary_emotion:"HOME_EMOTION"}]});
 if(url==="/api/clients/c/threads/timeline-fails")return response({messages:[{role:"ai",content:"TARGET_PARTIAL",id:"t",safety_level:"normal"}],emotion:null});
 if(url==="/api/clients/c/threads/timeline-fails/emotion-timeline?limit=5")return response({},500);
 if(url==="/api/clients/c/threads/snapshot-fails")return response({},500);
 if(url==="/api/clients/c/threads/snapshot-fails/emotion-timeline?limit=5")return response({timeline:[{primary_emotion:"SHOULD_NOT_RENDER"}]});
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"id"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
function assertHome(){assert.equal(store.get("hdu_erc_thread_id"),"home");assert.equal(elements["#messages"].children[0].children[0].textContent,"HOME");assert.ok(elements["#emotion-timeline"].children[0].textContent.includes("HOME_EMOTION"));assert.equal(elements["#safety-status"].hidden,false);assert.ok(elements["#safety-status"].textContent.includes("紧急"));}
setImmediate(()=>setImmediate(async()=>{try{assertHome();await assert.rejects(context.__HDU_ERC_TEST__.selectThread("timeline-fails"));assertHome();await assert.rejects(context.__HDU_ERC_TEST__.selectThread("snapshot-fails"));assertHome();}catch(e){console.error(e);process.exit(1);}}));
''')


def test_static_create_load_failure_keeps_new_catalog_entry_and_restores_old_view():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");const store=new Map([["hdu_erc_client_id","c"],["hdu_erc_thread_id","old"]]);
const fetch=async(url,options={})=>{
 if(url==="/api/clients/c/threads"&&!options.method)return response({threads:[{thread_id:"old",title:"旧"}]});
 if(url==="/api/clients/c/threads"&&options.method==="POST")return response({thread:{thread_id:"new",title:"新"}},201);
 if(url==="/api/clients/c/threads/old")return response({messages:[{role:"human",content:"旧内容",id:"h"}],emotion:null});
 if(url==="/api/clients/c/threads/old/emotion-timeline?limit=5")return response({timeline:[{primary_emotion:"旧情绪"}]});
 if(url==="/api/clients/c/threads/new")return response({},500);
 if(url==="/api/clients/c/threads/new/emotion-timeline?limit=5")return response({timeline:[]});
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"id"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(async()=>{try{await assert.rejects(context.__HDU_ERC_TEST__.createThread());assert.equal(store.get("hdu_erc_thread_id"),"old");assert.equal(elements["#messages"].children[0].children[0].textContent,"旧内容");const ids=elements["#thread-list"].children.map(li=>li.children[0].attributes["data-thread-id"]);assert.deepEqual(ids,["old","new"]);}catch(e){console.error(e);process.exit(1);}}));
''')


def test_static_committed_delete_list_failure_never_restores_deleted_thread():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");const store=new Map([["hdu_erc_client_id","c"],["hdu_erc_thread_id","deleted"]]);let deleted=false;
const fetch=async(url,options={})=>{
 if(url==="/api/clients/c/threads"&&!options.method)return deleted?response({},500):response({threads:[{thread_id:"deleted",title:"删"},{thread_id:"remain",title:"留"}]});
 if(url==="/api/clients/c/threads/deleted"&&options.method==="DELETE"){deleted=true;return response({},204);}
 if(url==="/api/clients/c/threads/deleted")return response({messages:[{role:"human",content:"DELETED_SECRET",id:"d"}],emotion:null});
 if(url==="/api/clients/c/threads/deleted/emotion-timeline?limit=5")return response({timeline:[]});
 if(url==="/api/clients/c/threads/remain")return response({messages:[{role:"human",content:"REMAIN",id:"r"}],emotion:null});
 if(url==="/api/clients/c/threads/remain/emotion-timeline?limit=5")return response({timeline:[]});
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"id"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(async()=>{try{await assert.rejects(context.__HDU_ERC_TEST__.deleteCurrentThread());assert.equal(store.get("hdu_erc_thread_id"),"remain");assert.equal(elements["#messages"].children[0].children[0].textContent,"REMAIN");const ids=elements["#thread-list"].children.map(li=>li.children[0].attributes["data-thread-id"]);assert.deepEqual(ids,["remain"]);assert.ok(!elements["#messages"].children.map(x=>x.children[0].textContent).join(" ").includes("DELETED"));}catch(e){console.error(e);process.exit(1);}}));
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
setImmediate(()=>setImmediate(async()=>{try{await assert.rejects(context.__HDU_ERC_TEST__.createThread());assert.equal(store.get("hdu_erc_thread_id"),"t");assert.equal(elements["#new-thread-button"].disabled,false);await assert.rejects(context.__HDU_ERC_TEST__.deleteCurrentThread());assert.equal(store.get("hdu_erc_thread_id"),undefined);assert.ok(elements["#app-status"].textContent.includes("重新同步")||elements["#app-status"].textContent.includes("刷新"));assert.equal(elements["#delete-thread-button"].disabled,false);}catch(e){console.error(e);process.exit(1);}}));
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
  {role:"ai",content:[],id:"ai_1",regenerated:true,original_content:"旧答案",regeneration_reason:"不准确",predicted_emotion:"sad"},
  {role:"ai",content:"危机回复",id:"ai_2",safety_level:"crisis"},
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


def test_static_profile_emotion_feedback_and_draft_use_scoped_urls_and_accessible_controls():
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
 if(url.endsWith("/emotion-feedback"))return response({status:"saved"},201);
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"id"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{activeElement:null,querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(async()=>{try{
  const message=elements["#messages"].children[0],controls=message.children[1];
  assert.equal(controls.children[0].attributes["aria-label"],"重新生成回复");
  controls.children[1].listeners.click();const emotionChoices=controls.children[2];await emotionChoices.children[0].listeners.click();const emotionFeedback=calls.find(c=>c.url.endsWith("/emotion-feedback"));assert.equal(emotionFeedback.options.method,"POST");assert.deepEqual(JSON.parse(emotionFeedback.options.body),{message_id:"a1",feedback:"accurate",predicted_emotion:"sad",turn_count:1});assert.ok(calls.some(c=>c.url.endsWith("/emotion-timeline?limit=5")));
  await elements["#profile-button"].listeners.click();await Promise.resolve();const editForm=elements["#profile-panel-body"].children[0];assert.equal(editForm.className,"profile-form");const preferred=editForm.children[0].children[0];preferred.value="新称呼";await editForm.listeners.submit({preventDefault(){}});const saved=calls.find(c=>c.options.method==="PUT");assert.deepEqual(JSON.parse(saved.options.body),{profile:{preferred_name:"新称呼"}});
  context.document.activeElement=elements["#profile-onboarding-start"];await elements["#profile-onboarding-start"].listeners.click();const questionForm=elements["#profile-panel-body"].children[0];questionForm.children[0].children[0].value="草稿称呼";await questionForm.listeners.submit({preventDefault(){}});const draft=calls.find(c=>c.url.endsWith("/profile/draft"));assert.equal(draft.options.method,"POST");assert.deepEqual(JSON.parse(draft.options.body),{thread_id:"t",answers:[{key:"preferred_name",answer:"草稿称呼"}]});assert.equal(elements["#profile-panel-body"].children[0].className,"profile-form");elements["#profile-close"].listeners.click();assert.equal(elements["#profile-button"].focused,true);assert.notEqual(elements["#profile-onboarding-start"].focused,true);
  assert.ok(profileGets>=1);
}catch(e){console.error(e);process.exit(1);}}));
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


def test_static_done_fallback_survives_refresh_select_and_create_failures():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");const store=new Map([["hdu_erc_client_id","c"],["hdu_erc_thread_id","home"]]);const encoder=new TextEncoder();let homeSnapshots=0;
function streamResponse(text){const bytes=encoder.encode(text);let read=false;return {ok:true,status:200,body:{getReader(){return {async read(){if(read)return {done:true};read=true;return {done:false,value:bytes};},releaseLock(){}};}}};}
const fetch=async(url,options={})=>{
 if(url==="/api/clients/c/threads"&&!options.method)return response({threads:[{thread_id:"home",title:"H"},{thread_id:"bad",title:"B"}]});
 if(url==="/api/clients/c/threads"&&options.method==="POST")return response({thread:{thread_id:"new-bad",title:"N"}},201);
 if(url==="/api/clients/c/threads/home"){homeSnapshots++;return homeSnapshots===1?response({messages:[{role:"ai",content:"BEFORE",id:"old",safety_level:"supportive"}],emotion:{primary_emotion:"before"}}):response({},500);}
 if(url==="/api/clients/c/threads/home/emotion-timeline?limit=5")return response({timeline:[{primary_emotion:"before-timeline"}]});
 if(url==="/api/clients/c/threads/bad"||url==="/api/clients/c/threads/new-bad")return response({},500);
 if(url.endsWith("/bad/emotion-timeline?limit=5")||url.endsWith("/new-bad/emotion-timeline?limit=5"))return response({timeline:[]});
 if(url.endsWith("/home/messages:stream"))return streamResponse('event: run_started\ndata: {"request_id":"turn-id","thread_id":"home"}\n\nevent: user_message\ndata: {"message_id":"human-new","role":"human","content":"HELLO"}\n\nevent: emotion_done\ndata: {"emotion":"hopeful","state":{"primary_emotion":"hopeful"}}\n\nevent: safety\ndata: {"level":"crisis","guidance":"private"}\n\nevent: token\ndata: {"content":"AFTER"}\n\nevent: done\ndata: {"message_id":"ai-new","content":"AFTER"}\n\n');
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"turn-id"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
function assertLatest(){assert.equal(store.get("hdu_erc_thread_id"),"home");assert.deepEqual(elements["#messages"].children.map(x=>x.children[0].textContent),["BEFORE","HELLO","AFTER"]);assert.ok(elements["#emotion-status"].textContent.includes("hopeful"));assert.equal(elements["#safety-status"].hidden,false);assert.ok(elements["#safety-status"].textContent.includes("紧急"));}
setImmediate(()=>setImmediate(async()=>{try{await context.__HDU_ERC_TEST__.streamMessage("HELLO");assertLatest();await assert.rejects(context.__HDU_ERC_TEST__.selectThread("bad"));assertLatest();await assert.rejects(context.__HDU_ERC_TEST__.createThread());assertLatest();const ids=elements["#thread-list"].children.map(li=>li.children[0].attributes["data-thread-id"]);assert.ok(ids.includes("new-bad"));}catch(e){console.error(e);process.exit(1);}}));
''')


def test_static_regeneration_done_fallback_restores_latest_same_id_after_select_failure():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");const store=new Map([["hdu_erc_client_id","c"],["hdu_erc_thread_id","home"]]);const encoder=new TextEncoder();let snapshots=0;
function streamResponse(text){const bytes=encoder.encode(text);let read=false;return {ok:true,status:200,body:{getReader(){return {async read(){if(read)return {done:true};read=true;return {done:false,value:bytes};},releaseLock(){}};}}};}
const fetch=async(url,options={})=>{
 if(url==="/api/clients/c/threads")return response({threads:[{thread_id:"home",title:"H"},{thread_id:"bad",title:"B"}]});
 if(url==="/api/clients/c/threads/home"){snapshots++;return snapshots===1?response({messages:[{role:"ai",content:"BEFORE",id:"a1",safety_level:"crisis",predicted_emotion:"sad"}],emotion:null}):response({},500);}
 if(url==="/api/clients/c/threads/home/emotion-timeline?limit=5")return response({timeline:[]});
 if(url==="/api/clients/c/threads/bad")return response({},500);
 if(url==="/api/clients/c/threads/bad/emotion-timeline?limit=5")return response({timeline:[]});
 if(url.endsWith("/messages/a1/regenerate:stream"))return streamResponse('event: run_started\ndata: {"request_id":"regen-id","thread_id":"home"}\n\nevent: token\ndata: {"content":"AFTER"}\n\nevent: done\ndata: {"message_id":"a1","content":"AFTER","reason":"不准确","regenerated":true}\n\n');
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"regen-id"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(async()=>{try{let wrapper=elements["#messages"].children[0],controls=wrapper.children[1],status=controls.children.at(-1);await context.__HDU_ERC_TEST__.submitRegeneration(wrapper,"a1","不准确",controls,status);wrapper=elements["#messages"].children[0];assert.equal(wrapper.attributes["data-message-id"],"a1");assert.equal(wrapper.children[0].textContent,"AFTER");assert.equal(wrapper.attributes["data-original-content"],"BEFORE");assert.equal(wrapper.attributes["data-regeneration-reason"],"不准确");await assert.rejects(context.__HDU_ERC_TEST__.selectThread("bad"));wrapper=elements["#messages"].children[0];assert.equal(wrapper.attributes["data-message-id"],"a1");assert.equal(wrapper.children[0].textContent,"AFTER");assert.equal(wrapper.attributes["data-original-content"],"BEFORE");assert.equal(wrapper.attributes["data-predicted-emotion"],"sad");assert.equal(elements["#safety-status"].hidden,false);}catch(e){console.error(e);process.exit(1);}}));
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


def test_static_pending_snapshot_retries_same_request_and_content():
    _run_node(NODE_DOM + r'''
const assert=require("assert"),fs=require("fs"),vm=require("vm");
const calls=[];const encoder=new TextEncoder();
function streamResponse(text){const bytes=encoder.encode(text);let sent=false;return {ok:true,status:200,body:{getReader(){return {async read(){if(sent)return {done:true};sent=true;return {done:false,value:bytes};},releaseLock(){}};}}};}
const fetch=async(url,options={})=>{calls.push({url,options});
 if(url.endsWith("messages:stream"))return streamResponse('event: run_started\ndata: {"request_id":"pending-id","operation":"turn","thread_id":"t"}\n\nevent: done\ndata: {"message_id":"ai_pending-id","content":"完成"}\n\n');
 return new Promise(()=>{});};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"new-id"},localStorage:{getItem:()=>null,setItem(){},removeItem(){}},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);context.__HDU_ERC_TEST__.setStateForTest("c","t");context.__HDU_ERC_TEST__.renderSnapshot({messages:[{role:"human",id:"human_pending-id",content:"原消息"}],emotion:null,pending_turn:{request_id:"pending-id",content:"原消息"}});
(async()=>{try{assert.equal(elements["#message-input"].disabled,true);assert.equal(elements["#pending-turn"].hidden,false);await elements["#pending-retry-button"].listeners.click();const request=calls.find(c=>c.url.endsWith("messages:stream"));assert.deepEqual(JSON.parse(request.options.body),{message:"原消息",request_id:"pending-id"});assert.equal(elements["#pending-turn"].hidden,true);assert.equal(elements["#message-input"].disabled,false);}catch(e){console.error(e);process.exit(1);}})();
''')


def test_static_app_uses_client_scoped_post_streams_and_server_only_state():
    app_js = (ROOT / "chatbot/static/app.js").read_text(encoding="utf-8")
    assert "messages:stream" in app_js and "regenerate:stream" in app_js
    assert "getReader()" in app_js and "TextDecoder" in app_js and "request_id" in app_js
    assert "new EventSource" not in app_js
    assert "/api/session" not in app_js
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
                ]
    for method, path, payload in requests:
        response = getattr(client, method)(path, json=payload) if payload is not None else getattr(client, method)(path)
        assert response.status_code == 404, path
