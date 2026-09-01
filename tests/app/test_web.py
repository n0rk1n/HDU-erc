import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chatbot.chat_service import ChatEvent
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
    assert format_sse(ChatEvent("token", {"content": "你好"})) == (
        'event: token\ndata: {"content": "你好"}\n\n'
    )


def test_static_assets_and_accessible_thread_controls_exist():
    index = (ROOT / "chatbot/static/index.html").read_text(encoding="utf-8")
    assert 'id="thread-list"' in index
    assert 'id="new-thread-button"' in index
    assert 'id="delete-thread-button"' in index
    assert 'aria-label="对话列表"' in index


NODE_DOM = r'''
class Element {
  constructor(name) {
    this.name=name; this.children=[]; this.parent=null; this.listeners={}; this.attributes={};
    this.textContent=""; this.className=""; this.value=""; this.disabled=false; this.hidden=false;
    this.scrollTop=0; this.scrollHeight=0;
  }
  appendChild(c){c.parent=this;this.children.push(c);return c;}
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
  {role:"system",content:{kind:"notice"},id:"sys"},
  {role:"tool",content:[{type:"image",url:"private"}],id:"tool"},
  {role:"ai",content:[],id:"ai_1",regenerated:true,original_content:"旧答案",regeneration_reason:"不准确",feedback:"like",predicted_emotion:"sad"},
]});
  assert.equal(elements["#messages"].children.length,3);
  const ai=elements["#messages"].children[2];
  assert.equal(ai.attributes["data-message-id"],"ai_1");
  assert.equal(ai.attributes["data-original-content"],"旧答案");
  assert.equal(ai.attributes["data-regeneration-reason"],"不准确");
  assert.equal(ai.attributes["data-predicted-emotion"],"sad");
  assert.equal(ai.children[0].textContent,"[工具调用]");
} catch(e) { console.error(e); process.exit(1); }
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
 if(url==="/api/clients/c/threads/t/messages:stream")return streamResponse('event: user_message\ndata: {"message_id":"h1","role":"human","content":"你好"}\n\nevent: token\ndata: {"content":"回"}\n\nevent: token\ndata: {"content":"复"}\n\nevent: safety\ndata: {"level":"supportive","guidance":"陪伴"}\n\nevent: done\ndata: {"message_id":"a1","content":"回复"}\n\n');
 throw new Error(`unexpected ${url}`);};
const context={console,fetch,encodeURIComponent,TextDecoder,TextEncoder,AbortController,crypto:{randomUUID:()=>"00000000-0000-4000-8000-000000000009"},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},sessionStorage:{getItem:()=>null,setItem(){}},document:{querySelector:s=>elements[s]||null,createElement:n=>new Element(n)},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync("chatbot/static/app.js","utf8"),context);
setImmediate(()=>setImmediate(async()=>{try{await context.__HDU_ERC_TEST__.streamMessage("你好");const request=calls.find(c=>c.url.endsWith("messages:stream"));assert.equal(request.options.method,"POST");assert.deepEqual(JSON.parse(request.options.body),{message:"你好",request_id:"00000000-0000-4000-8000-000000000009"});assert.deepEqual(elements["#messages"].children.map(x=>x.children[0].textContent),["你好","回复"]);assert.equal(elements["#safety-status"].textContent,"陪伴");assert.equal(elements["#send-button"].disabled,false);}catch(e){console.error(e);process.exit(1);}}));
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
 if(url==="/api/clients/c/threads/t/messages/a1/regenerate:stream")return streamResponse('event: token\ndata: {"content":"新"}\n\nevent: done\ndata: {"message_id":"a1","content":"新","reason":"不准确","regenerated":true}\n\n');
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
