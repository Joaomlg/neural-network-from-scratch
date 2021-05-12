import logging
logging.getLogger('werkzeug').disabled = True

from flask import Flask, render_template, request, make_response
from threading import Thread
import numpy as np
import json
import os
import webbrowser
from waitress import serve

class WebApp:
  def __init__(self, network: 'Network', port=8080):
    curr_dir = os.path.dirname(__file__)
    template_folder = os.path.join(curr_dir, 'templates')
    static_folder = os.path.join(curr_dir, 'static')

    self.app = Flask('WebApp', template_folder=template_folder, static_folder=static_folder)
    self.network = network
    self.host = '0.0.0.0'
    self.port = port
    self.server_url = f'http://localhost:{self.port}'

  def run(self):
    @self.app.route('/')
    def index():
      return render_template('index.html')
    
    @self.app.route('/predict', methods=['POST'])
    def predict():
      data = np.array(json.loads(request.data)).reshape((1, 1, 28, 28))
      result = self.network.predict(data)[0]
      num = int(np.argmax(result))
      prob = result[num]
      return make_response(json.dumps({'result': num, 'probability': prob}), 200)
    
    server = Thread(target=serve, args=[self.app], kwargs={'host': self.host, 'port': self.port})
    server.start()

    print(f'Serving web app on {self.server_url}')

    webbrowser.open_new_tab(self.server_url)