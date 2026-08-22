"""对外 HTTP 接口域：admin WebUI 的 FastAPI 应用与路由。"""

from .server import AdminServer, create_admin_app

__all__ = ["AdminServer", "create_admin_app"]
