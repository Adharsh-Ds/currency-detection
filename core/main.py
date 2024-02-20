import os
import re
import cv2
import numpy as np

from PIL import Image
from doctr.models import ocr_predictor



predictor = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)



def generate_tokens_mindee(given_image: Image.Image):
    cv_img = cv2.cvtColor(np.array(given_image.copy().convert("RGB")), cv2.COLOR_RGB2BGR)
    result = predictor([cv_img])
    json_export = result.export()
    page_words = [[word for block in page['blocks'] for line in block['lines'] for word in line['words']] for page in
                  json_export['pages']]
    page_dims = [page['dimensions'] for page in json_export['pages']]

    words_abs_coords = [
        [[[int(round(word['geometry'][0][0] * dims[1])), int(round(word['geometry'][0][1] * dims[0])),
           int(round(word['geometry'][1][0] * dims[1])), int(round(word['geometry'][1][1] * dims[0]))],
          word.get('value'), word.get('confidence')] for word in words]
        for words, dims in zip(page_words, page_dims)
    ]
    return words_abs_coords



def process_ocr_output(ocr_output):
    n = []
    n_plus_1 = []
    n_less_1 = []
    currency_symbols = ['$', '€', '£', '¥', '₹', '₣']
    symbol_str = ['AED', 'INR', 'USD', 'CAD', 'OMR', 'QAR', 'SAR','EUR']

    for sublist in ocr_output:
        for i, inner_list in enumerate(sublist):
            detected_symbol = inner_list[1]
            confidence_score = inner_list[2]

            for symbol in currency_symbols:
                if symbol in detected_symbol:
                    match = re.match(rf"({symbol})(.*)", detected_symbol)
                    if match:
                        currency_symbol = match.group(1)
                        numerical_value = match.group(2)
                        n.append({'bbox': inner_list[0], 'symbol': currency_symbol, 'value': numerical_value, 'confidence_score': confidence_score})
                    else:
                        n.append({'bbox': inner_list[0], 'symbol': symbol, 'value': None, 'confidence_score': confidence_score})
                    if i + 1 < len(sublist):
                        next_item = sublist[i + 1]
                        n_plus_1.append({'bbox': next_item[0], 'value': next_item[1], 'confidence_score': next_item[2]})
                    if i - 1 >= 0:
                        prev_item = sublist[i - 1]
                        n_less_1.append({'bbox': prev_item[0], 'value': prev_item[1], 'confidence_score': prev_item[2]})
                    break
            
            for symbol_str_item in symbol_str:
                if detected_symbol == symbol_str_item:
                    n.append({'bbox': inner_list[0], 'symbol': detected_symbol, 'value': None, 'confidence_score': confidence_score})
                    if i + 1 < len(sublist):
                        next_item = sublist[i + 1]
                        n_plus_1.append({'bbox': next_item[0], 'value': next_item[1], 'confidence_score': next_item[2]})
                    if i - 1 >= 0:
                        prev_item = sublist[i - 1]
                        n_less_1.append({'bbox': prev_item[0], 'value': prev_item[1], 'confidence_score': prev_item[2]})
                    break

    return n, n_plus_1, n_less_1,


def extract_values(n, n_plus_1, n_less_1):
    result_dict = {}
    pattern = r'^[$€£¥]?[0-9,.]+$'

    if n:
        j = n[0]
        symbol = j.get('symbol')
        result_dict['symbol'] = symbol

        if symbol:
            if j.get('value') and re.match(pattern, j['value']):
                result_dict['n'] = j['value']
                result_dict['confidence_score'] = j['confidence_score']
            elif n_plus_1 and n_plus_1[0].get('value') and re.match(pattern, n_plus_1[0]['value']):
                result_dict['n_plus_1'] = n_plus_1[0]['value']
                result_dict['confidence_score'] = n_plus_1[0]['confidence_score']
            elif n_less_1 and n_less_1[0].get('value') and re.match(pattern, n_less_1[0]['value']):
                result_dict['n_less_1'] = n_less_1[0]['value']
                result_dict['confidence_score'] = n_less_1[0]['confidence_score']
    else:
        result_dict['symbol'] = None  

    return result_dict, result_dict.get('symbol')  



def get_currency_ascii(detected_symbols):
    currency_mapping = {
        'AED': 'د',
        'INR': '₹',
        'OMR': '﷼',
        'QAR': 'س',
        'SAR': 'ق',
        'USD': '$',
        'CAD': '$',
        'EUR': '€',
        'EGP': '£',
    }
    if detected_symbols != "":
        for currency, symbol in currency_mapping.items():
            if detected_symbols == currency or detected_symbols == symbol:
                return ord(symbol)

    else:
        return None

