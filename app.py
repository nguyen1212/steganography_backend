from flask import jsonify, Flask, request, send_file
import os, cv2, converter
from werkzeug.utils import secure_filename

app = Flask(__name__)

path = os.getcwd()

@app.route("/api/v1/encrypt", methods=['POST'])
def encrypt():
  file = request.files['file']
  text = request.form['text']
  key = request.form['key']

  if file:
    filename = secure_filename(file.filename)
    filepath = f'{app.instance_path}/{filename}'
    file_name= filename.split('.')[0]

    file.save(filepath)

    img = cv2.imread(filepath, 1)

    bit = converter.toBinary(text)
    hsi = converter.rgbToHSI(img, img.shape)
    iPlane = hsi[:,:,2]
    secret_msg = converter.genMsg(bit, iPlane.shape)
    stego = converter.embed(hsi, secret_msg)
    newfile = converter.hsiToRGB(stego, stego.shape)
    newfile_path =f'{app.instance_path}/{file_name}_encrypted.png'
    cv2.imwrite(newfile_path, newfile)
    
    return send_file(newfile_path, mimetype='image/png')
  
  return jsonify()

@app.route("/api/v1/decrypt", methods=['POST'])
def decrypt():
  file = request.files['file']
  key = request.form['key']
  if file:
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.instance_path, filename)

    file.save(filepath)

    img = cv2.imread(filepath, 1)

    hsi = converter.rgbToHSI(img, img.shape)
    bit = converter.extract(hsi)
    msg = converter.toString(bit)

    response = jsonify({
      'text': msg
    })
    
    return response

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=40002)
