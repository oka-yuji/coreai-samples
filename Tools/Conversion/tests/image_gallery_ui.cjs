const fs = require("fs");
const vm = require("vm");
const template=fs.readFileSync(process.argv[2],"utf8");
const html=fs.readFileSync(process.argv[3],"utf8");
const payload=JSON.parse(html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const source=template.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
function element(){return {children:[],attributes:{},textContent:"",value:"",files:[],append(...v){this.children.push(...v)},replaceChildren(...v){this.children=v},setAttribute(k,v){this.attributes[k]=v},querySelector(){return element()},click(){}};}
function make(){
 const elements=new Map();
 const get=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id)};
 get("payload").textContent=JSON.stringify(payload);
 const saved=new Map();
 const context=vm.createContext({document:{getElementById:get,createElement:element},structuredClone,localStorage:{getItem:k=>saved.get(k)||null,setItem:(k,v)=>saved.set(k,v)},Blob,URL,setTimeout});
 vm.runInContext(source,context);
 return {get,context,saved,score:()=>vm.runInContext("ratings.cohorts.C.ratings[0].axes.instruction",context)};
}
function scores(score){const value=structuredClone(payload.ratings);value.cohorts.C.ratings[0].axes.instruction=score;return JSON.stringify(value);}
async function latestSelection(){
 const state=make(); let oldDone,newDone;
 state.get("resume").files=[{text:()=>new Promise(r=>oldDone=r)}];state.get("resume").value="old.json";
 const earlier=state.get("resume").onchange();
 state.get("resume").files=[{text:()=>new Promise(r=>newDone=r)}];state.get("resume").value="new.json";
 const newer=state.get("resume").onchange();
 newDone(scores(2));await newer;const afterLatest=state.score();
 oldDone(scores(0));await earlier;
 return {case:"last_selected_file_wins",afterLatest,afterStale:state.score(),passed:state.score()===2};
}
async function editDuringResume(){
 const state=make();let done;
 state.get("resume").files=[{text:()=>new Promise(r=>done=r)}];state.get("resume").value="old.json";
 const restore=state.get("resume").onchange();
 const card=state.get("columns").children[3];
 const control=card.children.slice(2).map(label=>label.children[1]).find(e=>e.attributes["aria-label"]==="C instruction");
 control.value="2";control.onchange();const afterEdit=state.score();
 done(scores(0));await restore;
 return {case:"manual_edit_invalidates_pending_restore",afterEdit,afterStale:state.score(),inputValue:state.get("resume").value,passed:state.score()===2};
}
(async()=>{const results=[await latestSelection(),await editDuringResume()];console.log(JSON.stringify({node:process.version,results},null,2));process.exitCode=results.every(row=>row.passed)?0:1;})().catch(error=>{console.error(error);process.exitCode=2});
