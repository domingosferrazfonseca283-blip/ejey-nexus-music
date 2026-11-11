const $=id=>document.getElementById(id);
const LIBRARY_KEY="ejey-nexus-library-v1",ENDPOINT_KEY="ejey-nexus-endpoint-v1",TOKEN_KEY="ejey-nexus-token-v1";
let currentPrompt="";
function readLibrary(){try{return JSON.parse(localStorage.getItem(LIBRARY_KEY)||"[]")}catch{return[]}}
function buildPrompt(){
 const idea=$("idea").value.trim();
 if(!idea){$("result").textContent="Escreve primeiro uma ideia musical.";return""}
 currentPrompt=[`Create an original ${$("genre").value} composition.`,`Mood: ${$("mood").value}. Tempo: ${$("tempo").value}. Vocal arrangement: ${$("vocals").value}.`,`Creative direction: ${idea}`,"Use an original melody and arrangement. Do not imitate a specific living artist or use recordings without permission."].join("\n");
 $("result").textContent=currentPrompt;return currentPrompt;
}
function renderLibrary(){
 const items=readLibrary();$("library").replaceChildren();
 if(!items.length){const p=document.createElement("p");p.textContent="A biblioteca está vazia.";$("library").append(p);return}
 items.forEach((item,index)=>{
  const card=document.createElement("div");card.className="track";
  const title=document.createElement("strong");title.textContent=item.title;
  const meta=document.createElement("span");meta.textContent=item.date+" · "+item.genre;
  const open=document.createElement("button");open.className="btn secondary";open.textContent="Abrir prompt";open.onclick=()=>{currentPrompt=item.prompt;$("result").textContent=item.prompt};
  const del=document.createElement("button");del.className="btn secondary";del.textContent="Apagar";del.onclick=()=>{const next=readLibrary();next.splice(index,1);localStorage.setItem(LIBRARY_KEY,JSON.stringify(next));renderLibrary()};
  card.append(title,meta,open,del);$("library").append(card);
 });
}
function baseUrl(){return $("endpoint").value.trim().replace(/\/+$/,"")}
async function api(path,options={}){
 const base=baseUrl();if(!base)throw Error("Configura o endereço do motor no PC.");
 const headers={...(options.headers||{})};if(options.body)headers["Content-Type"]="application/json";
 const token=$("apiToken").value.trim();if(token)headers.Authorization="Bearer "+token;
 const response=await fetch(base+path,{...options,headers,signal:AbortSignal.timeout(12000)});
 let data={};try{data=await response.json()}catch{data={detail:"Resposta inválida do servidor."}}
 return {response,data};
}
async function testConnection(){
 $("connect").disabled=true;$("connection").textContent="A testar ligação...";
 try{const {response,data}=await api("/health");if(!response.ok)throw Error(data.detail||"Erro HTTP "+response.status);
 $("connection").textContent="Ligado · API "+data.api_version+" · Motor: "+data.engine_status+". "+data.message;
 }catch(e){$("connection").textContent="Falha na ligação: "+e.message}
 finally{$("connect").disabled=false}
}
async function sendGeneration(){
 const prompt=currentPrompt||buildPrompt();if(!prompt)return;
 $("generate").disabled=true;$("generationStatus").textContent="A enviar pedido...";
 try{
  const {response,data}=await api("/v1/generate",{method:"POST",body:JSON.stringify({prompt,genre:$("genre").value,mood:$("mood").value,tempo:$("tempo").value,vocals:$("vocals").value})});
  $("generationStatus").textContent=data.message||data.detail||("Resposta HTTP "+response.status);
 }catch(e){$("generationStatus").textContent="Pedido não enviado: "+e.message}
 finally{$("generate").disabled=false}
}
$("compose").onclick=buildPrompt;
$("save").onclick=()=>{const prompt=currentPrompt||buildPrompt();if(!prompt)return;const items=readLibrary();items.unshift({title:$("idea").value.trim().slice(0,48)||"Nova ideia",genre:$("genre").value,prompt,date:new Date().toLocaleString("pt-PT")});localStorage.setItem(LIBRARY_KEY,JSON.stringify(items));renderLibrary();$("result").textContent="Ideia guardada neste dispositivo."};
$("connect").onclick=()=>{const endpoint=baseUrl();if(endpoint&&!/^https?:\/\//i.test(endpoint)){$("connection").textContent="Usa um endereço HTTP ou HTTPS válido.";return}localStorage.setItem(ENDPOINT_KEY,endpoint);localStorage.setItem(TOKEN_KEY,$("apiToken").value.trim());testConnection()};
$("generate").onclick=sendGeneration;
$("endpoint").value=localStorage.getItem(ENDPOINT_KEY)||"";$("apiToken").value=localStorage.getItem(TOKEN_KEY)||"";
renderLibrary();
if("serviceWorker" in navigator)window.addEventListener("load",()=>navigator.serviceWorker.register("./sw.js").catch(()=>{}));
