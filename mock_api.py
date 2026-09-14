"""Optional FastAPI adapter; importing this module does not require FastAPI."""
def create_app(client=None):
    try:
        from fastapi import FastAPI, Request
    except ImportError as exc:
        raise RuntimeError("FastAPI is optional; use MockBankClient in Streamlit Demo Mode") from exc
    from .bank_client import MockBankClient
    client = client or MockBankClient()
    app = FastAPI(title="Bank AI Demo Integration API")
    @app.get("/api/customers")
    def customers(): return client.list_customers()
    @app.get("/api/customers/{customer_id}")
    def customer(customer_id: str):
        value = client.get_customer(customer_id)
        return value or {"error": "not_found"}
    @app.post("/api/customers")
    def create(payload: dict): return client.create_customer(payload)
    @app.post("/api/sync")
    def sync(): return client.sync()
    @app.get("/api/health")
    def health(): return client.health()
    @app.get("/api/statistics")
    def statistics(): return client.statistics()
    @app.post("/api/webhooks/customer-created")
    async def customer_created_webhook(request: Request):
        return client.webhook(await request.json())

    # Keep the short legacy route for existing demos and clients.
    @app.post("/api/webhook")
    async def webhook(request: Request):
        return client.webhook(await request.json())
    return app
