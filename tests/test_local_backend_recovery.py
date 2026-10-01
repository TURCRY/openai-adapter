import asyncio, sys, types, unittest
from pathlib import Path
import httpx

class HTTPException(Exception):
    def __init__(self, status_code=500, detail=""):
        super().__init__(detail); self.status_code=status_code; self.detail=detail
class App:
    def __init__(self,*a,**k): self.state=types.SimpleNamespace()
    def add_middleware(self,*a,**k): pass
    def on_event(self,*a,**k): return lambda f:f
    def get(self,*a,**k): return lambda f:f
    def post(self,*a,**k): return lambda f:f
def ident(default=None,*a,**k): return default
fastapi=types.ModuleType("fastapi"); fastapi.FastAPI=App; fastapi.Header=ident; fastapi.File=ident; fastapi.Form=ident; fastapi.Body=ident; fastapi.UploadFile=object; fastapi.Request=object; fastapi.HTTPException=HTTPException
sys.modules.setdefault("fastapi", fastapi); sys.modules.setdefault("fastapi.middleware", types.ModuleType("fastapi.middleware"))
cors=types.ModuleType("fastapi.middleware.cors"); cors.CORSMiddleware=object; sys.modules.setdefault("fastapi.middleware.cors", cors)
resp=types.ModuleType("fastapi.responses"); resp.Response=object; resp.JSONResponse=object; resp.StreamingResponse=object; sys.modules.setdefault("fastapi.responses", resp)
if "pydantic" not in sys.modules:
    p=types.ModuleType("pydantic"); p.BaseModel=type("BaseModel", (), {}); sys.modules["pydantic"]=p
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import adapter

class FakeHTTP:
    def __init__(self, outcomes): self.outcomes=list(outcomes); self.calls=[]
    async def request(self, method, url, **kwargs):
        self.calls.append((method,url,kwargs)); out=self.outcomes.pop(0)
        if isinstance(out, Exception): raise out
        return out
def response(code, url, payload=None): return httpx.Response(code, json=payload or {}, request=httpx.Request("POST", url))

class Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.old=(adapter.LOCAL_BASE, list(adapter.LOCAL_BACKEND_CANDIDATES), adapter._http, adapter._probe_backend_ping, adapter.select_backend_url, adapter._last_ping_ok_ts, adapter._ensure_local_ready)
        adapter._last_ping_ok_ts=0.0; adapter._backend_reselect_lock=asyncio.Lock()
        async def ready(): return True
        adapter._ensure_local_ready=ready
    def tearDown(self):
        adapter.LOCAL_BASE, adapter.LOCAL_BACKEND_CANDIDATES, adapter._http, adapter._probe_backend_ping, adapter.select_backend_url, adapter._last_ping_ok_ts, adapter._ensure_local_ready = self.old
        adapter._backend_reselect_lock=asyncio.Lock()
    async def test_primary_available_used_directly(self):
        adapter.LOCAL_BASE="http://10.0.1.10:5050"; adapter.LOCAL_BACKEND_CANDIDATES=["http://10.0.1.10:5050","http://192.168.0.155:5050"]
        adapter._http=FakeHTTP([response(200,"http://10.0.1.10:5050/chat_openai",{"ok":1})])
        r=await adapter._local_request_once_with_runtime_fallback("POST","/chat_openai",timeout=1,json={},headers={})
        self.assertEqual(r.json()["ok"],1); self.assertEqual([c[1] for c in adapter._http.calls],["http://10.0.1.10:5050/chat_openai"])
    async def test_fallback_then_primary_recovery_not_sticky(self):
        adapter.LOCAL_BASE="http://192.168.0.155:5050"; adapter.LOCAL_BACKEND_CANDIDATES=["http://10.0.1.10:5050","http://192.168.0.155:5050"]
        adapter._http=FakeHTTP([response(200,"http://10.0.1.10:5050/chat_openai",{"ok":1})]); calls=[]
        async def probe(base,ping,headers=None): calls.append((base,ping)); return True,"HTTP 200"
        adapter._probe_backend_ping=probe
        await adapter._local_request_once_with_runtime_fallback("POST","/chat_openai",timeout=1,json={},headers={})
        self.assertEqual(adapter.LOCAL_BASE,"http://10.0.1.10:5050"); self.assertEqual(calls,[("http://10.0.1.10:5050",adapter.LOCAL_PING_PATH)]); self.assertEqual([c[1] for c in adapter._http.calls],["http://10.0.1.10:5050/chat_openai"])
    async def test_primary_down_can_fallback(self):
        adapter.LOCAL_BASE="http://10.0.1.10:5050"; adapter.LOCAL_BACKEND_CANDIDATES=["http://10.0.1.10:5050","http://192.168.0.155:5050"]
        req=httpx.Request("POST","http://10.0.1.10:5050/chat_openai"); adapter._http=FakeHTTP([httpx.ConnectError("down",request=req), response(200,"http://192.168.0.155:5050/chat_openai",{"ok":1})])
        async def select(candidates,ping_path,headers,probe): return "http://192.168.0.155:5050",[]
        adapter.select_backend_url=select
        await adapter._local_request_once_with_runtime_fallback("POST","/chat_openai",timeout=1,json={},headers={})
        self.assertEqual(adapter.LOCAL_BASE,"http://192.168.0.155:5050")
    async def test_local_gemma_4_alias_and_route(self):
        payload=adapter._local_openai_chat_payload(requested_model="local-gemma-4",messages=[{"role":"user","content":"ping"}],max_tokens=1,stream=False)
        self.assertEqual(payload["model"],"Gemma_4_12B_It_Q4_K_M")
        adapter.LOCAL_BASE="http://10.0.1.10:5050"; adapter.LOCAL_BACKEND_CANDIDATES=["http://10.0.1.10:5050","http://192.168.0.155:5050"]
        adapter._http=FakeHTTP([response(200,"http://10.0.1.10:5050/chat_openai",{"choices":[{"message":{"content":"OK"}}]})])
        await adapter._local_chat_openai(requested_model="local-gemma-4",messages=[{"role":"user","content":"ping"}],max_tokens=1,stream=False)
        self.assertEqual(adapter._http.calls[0][1],"http://10.0.1.10:5050/chat_openai"); self.assertEqual(adapter._http.calls[0][2]["json"]["model"],"Gemma_4_12B_It_Q4_K_M")
if __name__ == "__main__": unittest.main()

