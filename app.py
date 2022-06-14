from flask import jsonify, Flask, request
import pickle, json
import numpy as np
import __main__
from model import IdentityPassthrough, columnDropperTransformer

app = Flask(__name__)

__main__.IdentityPassthrough = IdentityPassthrough
__main__.columnDropperTransformer = columnDropperTransformer

stacked_model = pickle.load(open('model.pkl', 'rb'))
imputer_model = pickle.load(open('imputer_model.pkl', 'rb'))

# cors = CORS(app, resources={r"/*": {"origins": "https://tranquil-thicket-00848.herokuapp.com"}})

def classify_input(input):
  if input == 2:
    return np.nan
  return input

def threshold(input):
  i = 0
  for v in input[0]:
    if v > 0.6:
      input[0][i] = 1
    elif v > 0:
      if v < 0.4:
        input[0][i] = -1
      else:
        input[0][i] = 0
    i+=1
  return input


@app.route("/api/v1/predict", methods=['POST'])
def predict():
    url_stats = json.loads(request.data)
    datapoint = np.array([[
      classify_input(url_stats['having_ip_address']),
      classify_input(url_stats['url_length']),
      classify_input(url_stats['shortining_service']),
      classify_input(url_stats['having_at_symbol']),
      classify_input(url_stats['double_slash_redirecting']),
      classify_input(url_stats['prefix_suffix']),
      classify_input(url_stats['having_sub_domain']),
      classify_input(url_stats['ssl_final_state']),
      classify_input(url_stats['domain_registeration_length']),
      classify_input(url_stats['favicon']),
      classify_input(url_stats['port']),
      classify_input(url_stats['https_token']),
      classify_input(url_stats['request_url']),
      classify_input(url_stats['url_of_anchor']),
      classify_input(url_stats['links_in_tags']),
      classify_input(url_stats['sfh']),
      classify_input(url_stats['submitting_to_email']),
      classify_input(url_stats['abnormal_url']),
      classify_input(url_stats['redirect']),
      classify_input(url_stats['on_mouseover']),
      classify_input(url_stats['right_click']),
      classify_input(url_stats['pop_up_window']),
      classify_input(url_stats['iframe']),
      classify_input(url_stats['age_of_domain']),
      classify_input(url_stats['dns_record']),
      classify_input(url_stats['web_traffic']),
      classify_input(url_stats['page_rank']),
      classify_input(url_stats['google_index']),
      classify_input(url_stats['links_pointing_to_page']),
      classify_input(url_stats['statistical_report']),  
    ]])

    imputed_datapoint = imputer_model.transform(datapoint)
    threshold_datapoint = threshold(imputed_datapoint)
    result = stacked_model.predict_proba(threshold_datapoint).flatten()
    confidence = round(result[1],2)
    if confidence < 0.7 and confidence > 0.2:
      confidence -= 0.2
    is_phishing = bool(confidence < 0.5)

    response = jsonify({
      'confidence': confidence,
      'is_phishing': is_phishing,
    })
    
    return response

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=40002)
