from wsgiref.simple_server import make_server

import falcon
import json
import cv2
import os
import numpy as np
import utils

PORT = 7777
INPUT_VIDEO_FPATH = 'data/2018-02-2715_03_24.ogv'

html_body_template = '''<html>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>Mug Detector</title>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
</style>
<body>
%s
</body></html>
'''

class MugResource:
    def on_get(self, req, resp):
        print(f'Ensuring {INPUT_VIDEO_FPATH} was processed...')
        utils.ensure_video_precessed(INPUT_VIDEO_FPATH)
        
        resp.content_type = 'text/html'
        images, captions = utils.get_switches(INPUT_VIDEO_FPATH)
        images_with_captions = ""
        for im, cap in zip(images, captions):
            images_with_captions += f"""<p>"{cap}"</p>
                                        <img src="image/{im}">"""
        resp.body = html_body_template % images_with_captions
        resp.status = falcon.HTTP_200

def get_app():
    app = falcon.API()
    app.add_route('/', MugResource())
    app.add_static_route('/image', os.path.abspath('./') + '/')
    return app

app = get_app()

if __name__ == '__main__':
    with make_server('', PORT, app) as httpd:
        print(f'Serving on port {PORT}...')
        httpd.serve_forever()
