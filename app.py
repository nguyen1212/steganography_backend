from flask import jsonify, Flask, request, send_file, Response, json
from flask_cors import CORS
import os, cv2, converter, encryptor
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

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

    encrypted_text = encryptor.encryptAES(key, text)

    img = cv2.imread(filepath, 1)
    
    bit = converter.toBinary(encrypted_text)
    hsi = converter.rgbToHSI(img, img.shape)
    
    secret_msg = converter.genMsg(bit)
    if len(secret_msg) > 0.4 * img.shape[0] * img.shape[1]:
      return Response(
        "cover image is not enough",
        status=400,
      )

    stego, brokenPixelIndexList, pixelIndexList = converter.embed(hsi, secret_msg)
    newfile = converter.setFlag(stego, brokenPixelIndexList, pixelIndexList)
    newfile_path =f'{app.instance_path}/{file_name}_encrypted.png'
    cv2.imwrite(newfile_path, newfile)
    
    return send_file(newfile_path, mimetype='image/png')
  
  return Response(
    "unsupported file",
    status=400,
  )

@app.route("/api/v1/test", methods=["POST"])
def test():
  key = request.form['key']
  text = request.form['text']

  return jsonify({
    'text': text,
    'key': key
  })

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
    bit = converter.extract(img[:,:,0], hsi)
    msg = converter.toString(bit)

    decrypted_text = encryptor.decryptAES(key, msg)

    if decrypted_text != None:
      response = jsonify({
        'text': decrypted_text
      })
    else:
      response = Response(
        "We can't decrypt any message with the key provided",
        status=400,
      )
    
    return response
  
  return Response(
    "unsupported file",
    status=400,
  )

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=40002)
