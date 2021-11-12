"""
HTTP service for interacting with model. See README.md for details.

Supported requests:

- GET /labels
    Parameters: -
    Return: JSON with an array of labels available for the model

- POST /predict
    Parameters: JSON with the following structure
        {
            'description': some text for model to analyze
        }
    Return: JSON with the following structure
        {
            'description': text from the request,
            'prediction': predicted label as text
        }
"""

import logging as lg

from aiohttp import web

from src.util import MODEL_FOLDER, Model, set_logging

routes = web.RouteTableDef()
model = Model(
    f'{MODEL_FOLDER}/classifier_0',
    f'{MODEL_FOLDER}/labels.json'
)


@routes.get('/labels')
async def gel_labels(request):
    """
    Handler function for the GET /labels endpoint.

    :param request: aiohttp.web.BaseRequest
    :return: JSON
    """
    data = {
        'labels': model.labels
    }
    return web.json_response(
        data,
        status=200,
        content_type='application/json')


@routes.post('/predict')
async def get_prediction(request):
    """
    Handler function for the POST /predict endpoint.

    :param request: aiohttp.web.BaseRequest
    :return: JSON
    """
    if not request.body_exists:
        raise web.HTTPBadRequest
    req_data = await request.json()
    desc = req_data.get('description')
    data = {
        'description': desc,
        'prediction': model.get_prediction(desc)
    }
    lg.info(f'Prediction generated: {data}')
    return web.json_response(
        data,
        status=200,
        content_type='application/json')


# Main loop

if __name__ == '__main__':
    lg.info('Starting server')
    set_logging()
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app)
