import os
from aiohttp import web
import aiohttp_cors

appdir = os.path.join(
    os.path.dirname(__file__), 'ui', 'dist'
)

def get_app(*, dataset, image_dir=None):
    if image_dir is None:
        image_dir = os.path.dirname(dataset)

    async def index(_):
        return web.FileResponse(os.path.join(appdir, "index.html"))

    async def data(_):
        return web.FileResponse(dataset)

    async def image(request):
        name = request.match_info["name"]
        return  web.FileResponse(os.path.join(image_dir, os.path.basename(name)))

    @web.middleware
    async def spa_middleware(request, handler):
        try:
            return await handler(request)
        except web.HTTPException as ex:
            print(ex)
            if ex.status == 404:
                return await index(request)
            raise

    app = web.Application()
    app.add_routes(
        [
            web.get("/api/data", data),
            web.get("/api/image/{name}", image),
            web.get("/", index),
            web.static("/", appdir),
            web.get("", index),
        ]
    )

    cors = aiohttp_cors.setup(
        app,
        defaults={"*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*")},
    )
    for route in list(app.router.routes()):
        cors.add(route)

    app.middlewares.append(spa_middleware)

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="JSONL file containing dataset records")
    parser.add_argument('--image_dir', '-i', help="Location of images. If not set will look at the directory where dataset is located")
    parser.add_argument('--port', type=int, default=8000, help='Port to run server on')

    args = parser.parse_args()
    app = get_app(dataset=args.dataset, image_dir=args.image_dir)
    web.run_app(app, port=args.port)

