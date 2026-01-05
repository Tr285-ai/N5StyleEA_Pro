# Minimal ASGI app stub used only to satisfy conftest import during integration tests.
# The actual test suite does not use the test_client fixture.
async def app(scope, receive, send):
    if scope.get('type') == 'lifespan':
        while True:
            message = await receive()
            if message['type'] == 'lifespan.startup':
                await send({'type': 'lifespan.startup.complete'})
            elif message['type'] == 'lifespan.shutdown':
                await send({'type': 'lifespan.shutdown.complete'})
                return
    elif scope.get('type') == 'http':
        await send({
            'type': 'http.response.start',
            'status': 200,
            'headers': [(b'content-type', b'text/plain')],
        })
        await send({'type': 'http.response.body', 'body': b'OK'})
    else:
        # No-op for other scopes
        pass
