import cv2
import copy
import os
import re
from common_global_model import (
    filter_image_by_zone,
    page_no,
)
from PIL import Image
from fuzzywuzzy import fuzz
from table_processing import TableProcessing
from config import LOG_BOLIERPLATE
from common_helper import update_log_data
from common_helper import generate_log_json
from data_extraction import DocumentProcessing, reformed_lines_dict_data
from dateparser.search import search_dates
from common_helper import connect_aws_s3, connect_s3_resource
from config import (
    GLOBAL_MODEL_PATH,
    DRIVE_TMP_PATH,
    GLOBAL_MODEL_CONFIDENCE_SCORE_THRESHOLD,
    PROCESS_ASSETS_S3_BUCKET,
)
from yolo_prediction import model_pred
from cvu.detector.yolov5 import Yolov5 as Yolov5Onnx


class Cheque:
    def __init__(self):
        pass

    def init_document(
        self,
        document,
        document_pages,
        attributes,
        ocr_output_by_page,
        batch,
        classification_result,
        logapi,
        label_map,
        labels,
        layoutlm_model_path,
        layoutlm_model,
        device,
        classification_status,
        ocr_engine,
    ):
        self.batch = batch
        self.batch_process = self.batch.get("process", {})
        self.document = document
        self.document_pages = document_pages
        self.ocr_output_by_page = ocr_output_by_page
        self.attributes = attributes
        self.key_attributes = attributes.get("key_value", {})
        self.line_attributes = attributes.get("line_item", {})
        self.key_items = dict.fromkeys(self.key_attributes.keys(), "")
        self.key_items_cs = dict.fromkeys(self.key_attributes.keys(), 0)
        self.line_items = []
        self.line_items_cs = []
        self.classification_result = classification_result
        self.quality_score = {}
        self.derived_attributes = {}
        self.raw_texts = {}
        self.doctr_model = None
        self.doc_log = {}
        self.logapi = logapi
        self.label_map = label_map
        self.labels = labels
        self.layoutlm_model = layoutlm_model
        self.layoutlm_model_path = layoutlm_model_path
        self.device = device
        self.classification_status = classification_status
        self.pg_num_doc_list = []
        self.ocr_engine = ocr_engine
        self.pg_num_doc_ids = [0]
        self.s3_client = connect_aws_s3()
        self.s3_resource = connect_s3_resource()

    def global_model(self):
        try:
            folder_model = os.path.join(GLOBAL_MODEL_PATH, "Cheque")
            local_path = os.path.join(DRIVE_TMP_PATH, GLOBAL_MODEL_PATH, "Cheque")
            file_names = ["Cheque.onnx"]
            for file_name in file_names:
                s3_key = os.path.join(folder_model, file_name)
                status = self.s3_client.head_object(
                    Bucket=PROCESS_ASSETS_S3_BUCKET, Key=s3_key
                )
                # The object exist.
                if status["ResponseMetadata"]["HTTPStatusCode"] != 404:
                    # create log path folder in tmp forlder
                    if not os.path.exists(local_path):
                        os.makedirs(local_path)
                    # download model weight file to tmp
                    tmp_path = os.path.join(local_path, file_name)
                    if not os.path.exists(tmp_path):
                        print("DOWNLOADING ::: ", tmp_path)
                        self.s3_client.download_file(
                            Bucket=PROCESS_ASSETS_S3_BUCKET,
                            Key=s3_key,
                            Filename=tmp_path,
                        )
                # The object does not exist
                else:
                    print("RETURNING FALSE")
                    return False
            return tmp_path
        except Exception as e:
            print(" Model Download  ::: Exception ::: ", e)
        return False

    def mapping(self, Key_item):
        Mapped_data = {}
        cords = (0, 0, 0, 0)
        for attribute_name in self.key_attributes:
            data_name = self.key_attributes[attribute_name].get("model_attribute_id")
            page_num = None
            if self.key_attributes[attribute_name].get("attribute_page_num"):
                page_num = self.key_attributes[attribute_name].get("attribute_page_num")
            regex = None
            zone = self.key_attributes[attribute_name].get("attribute_zone")
            if self.key_attributes[attribute_name].get("attribute_regex") != "":
                regex = eval(self.key_attributes[attribute_name].get("attribute_regex"))

            if data_name:
                data_name = data_name.lower()
                flag_pg = False
                if page_num:
                    page_number = page_no(page_num, Key_item)
                    for pg_no in page_number:
                        if (
                            pg_no in Key_item
                            and data_name in Key_item[pg_no][0]
                            and Key_item[pg_no][0][data_name]
                        ):
                            image_number = pg_no - 1
                            flag_pg = True
                            zone_type = Key_item[pg_no][0][data_name][0][4]
                            attribute_score = Key_item[pg_no][0][data_name][0][2]
                            score = attribute_score
                            if zone_type == "Invalid":
                                score -= GLOBAL_MODEL_CONFIDENCE_SCORE_THRESHOLD[
                                    "attribute_zoning"
                                ]

                            value = Key_item[pg_no][0][data_name][0][0]
                            if regex:
                                regex_rule = regex[0]
                                regex_patt = re.compile(regex_rule)
                                matched = re.search(regex_patt, value)
                                if matched:
                                    value = matched.group(0).strip()
                                else:
                                    score -= GLOBAL_MODEL_CONFIDENCE_SCORE_THRESHOLD[
                                        "regex"
                                    ]
                                    value = value
                                # matched = re.findall(regex_patt,value)
                                # value = " ".join([i for i in matched])
                            else:
                                value = value
                            cords = Key_item[pg_no][0][data_name][0][1]
                            score = score
                            Mapped_data.update(
                                {
                                    attribute_name: [
                                        (value, score, cords, zone, regex, image_number)
                                    ]
                                }
                            )

                if flag_pg is False:
                    flag = False
                    for key_value in Key_item:
                        if (
                            data_name in Key_item[key_value][0]
                            and Key_item[key_value][0][data_name]
                        ):
                            image_number = key_value - 1
                            flag = True
                            zone_type = Key_item[key_value][0][data_name][0][4]
                            attribute_score = Key_item[key_value][0][data_name][0][2]
                            score = attribute_score
                            if zone_type == "Invalid":
                                score -= GLOBAL_MODEL_CONFIDENCE_SCORE_THRESHOLD[
                                    "attribute_zoning"
                                ]

                            if page_num:
                                score -= GLOBAL_MODEL_CONFIDENCE_SCORE_THRESHOLD["page"]

                            value = Key_item[key_value][0][data_name][0][0]
                            if regex:
                                regex_rule = regex[0]
                                regex_patt = re.compile(regex_rule)
                                matched = re.search(regex_patt, value)
                                if matched:
                                    value = matched.group(0).strip()
                                else:
                                    value = value
                            else:
                                value = value
                            cords = Key_item[key_value][0][data_name][0][1]
                            score = score
                            Mapped_data.update(
                                {
                                    attribute_name: [
                                        (value, score, cords, zone, regex, image_number)
                                    ]
                                }
                            )
                    else:
                        if flag is False:
                            Mapped_data.update(
                                {attribute_name: [("", 10, cords, zone, regex, 0)]}
                            )
            else:
                key_val = DocumentProcessing()
                key_val.init_document(
                    self.document,
                    self.document_pages,
                    self.attributes,
                    self.ocr_output_by_page,
                    self.batch,
                    self.classification_result,
                    self.logapi,
                    self.label_map,
                    self.labels,
                    self.layoutlm_model_path,
                    self.layoutlm_model,
                    self.device,
                    self.classification_status,
                    self.ocr_engine,
                )
                key_val.process_document()
        return Mapped_data

    def directional_text_formation(
        self,
        page_document,
        ngrams,
        words,
        index,
        label_direction="right",
    ):
        radius_words = []
        duplicate_check = []
        matched_label = ngrams[index]
        radius_score = 0
        radius_cords = (None, None, None, None)
        image = Image.open(page_document)
        img_width, img_height = image.size
        max_word_x = max([word[0][2] for word in words])
        try:
            if label_direction.lower() == "bottom":
                line_text = ""
                right = min(matched_label[0][2] + int(max_word_x * 0.13), img_width)
                formed_lines = reformed_lines_dict_data(words[index:])
                idx = 0
                for line_idx, line_no in enumerate(formed_lines):
                    if idx > 2:
                        break
                    # flag set to true if the line words fall inside the right region
                    flag = False
                    word_count = 0
                    for word in formed_lines[line_no]:
                        if word_count > 4:
                            break
                        # if word falls on the matched_label
                        if line_idx <= 1:
                            if word[1] not in matched_label[1] and word[0][0] in range(
                                matched_label[0][0], matched_label[0][2]
                            ):
                                if word:
                                    word_count += 1
                                    duplicate_check.append(word[0])
                                    line_text += word[1] + " "
                                    radius_words.append(word)
                            elif word[1] not in matched_label[1] and matched_label[0][
                                0
                            ] in range(word[0][0], word[0][2]):
                                if word:
                                    word_count += 1
                                    duplicate_check.append(word[0])
                                    line_text += word[1] + " "
                                    radius_words.append(word)
                        # if word falls below the matched_label
                        if line_idx != 0:
                            if (
                                word[0][1] >= matched_label[0][3]
                                and word[0][2] > matched_label[0][0]
                                and word[0][2] < right
                                and word[0][1]
                                < matched_label[0][3]
                                + (5 * (matched_label[0][3] - matched_label[0][1]))
                            ):
                                if duplicate_check:
                                    if word[0] not in duplicate_check:
                                        if word:
                                            word_count += 1
                                            line_text += word[1] + " "
                                            radius_words.append(word)
                                else:
                                    if word:
                                        word_count += 1
                                        line_text += word[1] + " "
                                        radius_words.append(word)
                                flag = True
                    if flag:
                        idx += 1
            elif label_direction.lower() == "top":
                line_val_top = 3
                line_text = ""
                formed_lines = reformed_lines_dict_data(words[:index])
                top_idx = 0
                for line_idx, line_no in reversed(list(enumerate(formed_lines))):
                    if line_idx == 0:
                        continue
                    if top_idx >= line_val_top:
                        break
                    top_flag = False
                    left_word_count = 0
                    for word in formed_lines[line_no]:
                        if left_word_count > 4:
                            break
                        if (matched_label[0][0]) <= (word[0][2]) and (
                            matched_label[0][3]
                        ) >= word[0][3]:
                            if word:
                                top_flag = True
                                left_word_count = left_word_count + 1
                                line_text = line_text + word[1] + " "
                                radius_words.append(word)
                    if top_flag is True:
                        top_idx = top_idx + 1
            else:
                formed_line = []
                for word in words[index:]:
                    if (
                        word[0][0] > matched_label[0][2]
                        and word[0][2] > matched_label[0][2]
                        and word[0][1] < matched_label[0][3]
                        and word[0][3] > matched_label[0][1]
                    ):
                        formed_line.append(word)

                line_text = " ".join([i[1] for i in formed_line[:10]])
                radius_words.extend(formed_line)
            if radius_words:
                radius_score = sum([i[2] for i in radius_words]) / len(
                    [i[2] for i in radius_words]
                )
                words_cords = [i for i in zip(*[i[0] for i in radius_words])]
                left = min(words_cords[0])
                top = min(words_cords[1])
                right = max(words_cords[2])
                bottom = max(words_cords[3])
                radius_cords = (left, top, right, bottom)
        except Exception as e:
            print("directional_text_formation :: Exception :: ", e)
        return (
            line_text.replace(matched_label[1], "").strip(),
            radius_words,
            radius_score,
            matched_label,
            radius_cords,
        )

    def match_date_data(self, labels, regex, ocr_words, page_document):
        key_val = DocumentProcessing()
        label_matched = []
        for label_dict in labels:
            label = label_dict.get("label", "")
            label_direction = label_dict.get("direction", "").strip()
            label_len = len(str(label).split())
            ngram_key = "ngrams_" + str(label_len) if label_len > 1 else "words"
            if ngram_key != "words":
                words = key_val.generate_ngrams(ocr_words, n_range=label_len)
            else:
                words = ocr_words

            matched_lines = {}
            ngrams = words
            for index, ngram in enumerate(ngrams):
                # check if line includes valid label keywords
                line_text = ngram[1]
                label_matching_score = fuzz.ratio(label.lower(), line_text.lower())
                if label_matching_score > 85:
                    matched_lines[index] = label_matching_score
                    label_matched.append(ngram)

            # sort matched lines with score and iterate to match value
            matched_lines = dict(
                sorted(matched_lines.items(), key=lambda item: item[1], reverse=True)
            )

            if matched_lines:
                for index, label_score in matched_lines.items():
                    if label_direction:
                        (
                            radius_text,
                            radius_words,
                            radius_score,
                            matched_label,
                            radius_cords,
                        ) = self.directional_text_formation(
                            page_document,
                            ngrams,
                            words,
                            index,
                            label_direction,
                        )

                no_punct = radius_text
                if regex and no_punct:
                    if matched_label == "DDMMYYYY":
                        regex = "[0-9]{2}[0-9]{2}[0-9]{4}"
                    for regex_rule in regex:
                        regex_patt = re.compile("\\s" + regex_rule + "\\s")
                        matched = re.search(regex_patt, " " + no_punct + " ")
                        if matched:
                            matched_word = matched.group(0).strip()
                            return matched_word, radius_score, radius_cords
        return None, None, None

    def match_acc_routing_data(self, ocr_words):
        blacklisted_chr = ["⑆", "⑈", "-"]
        txt = ""
        routing = ""
        account_no = ""
        for words in ocr_words:
            txt += words[1] + " "
        for bl_ch in blacklisted_chr:
            txt = txt.replace(bl_ch, " ")
        data = re.findall("[0-9]+", txt)
        if len(data) > 3:
            data = data[-3:]
            meta_data = {}
            for idx, actual_data in enumerate(data):
                match = re.search("(?!0)", actual_data)
                zero_position = match.start()
                processed_data = actual_data[zero_position:]
                text_len = len(processed_data)
                meta_data.update(
                    {
                        idx: {
                            "Zero_count": zero_position,
                            "text": processed_data,
                            "len_text": text_len,
                            "data": actual_data,
                        }
                    }
                )

            cheque_value = 100
            cheque_no = []
            for i in meta_data:
                txt_len = meta_data[i]["len_text"]
                if txt_len < cheque_value:
                    cheque_no.clear()
                    cheque_value = txt_len
                    cheque_no.append(i)

            if cheque_no[0] == 0:
                routing = meta_data[1]["data"]
                account_no = meta_data[2]["data"]
                if len(account_no) < 5:
                    account_no = ""

            elif cheque_no[0] == 2:
                if meta_data[0]["data"][0] == "0":
                    routing = meta_data[0]["data"]
                    account_no = meta_data[1]["data"]
                    if len(account_no) < 5:
                        account_no = ""
                elif meta_data[1]["data"][0] == "0":
                    routing = meta_data[1]["data"]
                    account_no = meta_data[0]["data"]
                    if len(account_no) < 5:
                        account_no = ""
                else:
                    routing = meta_data[0]["data"]
                    account_no = meta_data[1]["data"]
                    if len(account_no) < 5:
                        account_no = ""
            elif cheque_no[0] == 1:
                routing = ""
                account_no = ""

            data = {"payoraccountno_data": account_no, "routingno_data": routing}

        return {"payoraccountno_data": "", "routingno_data": ""}

    def detect_text(self, model_prediction, ocr_words, zone_conf, page_no):
        data_fetch = {}
        for prediction in model_prediction:
            if "label" not in prediction[1] and prediction[1] not in data_fetch:
                logic_cs = 60
                zone_type = (
                    zone_conf[prediction[1]][0] if zone_conf[prediction[1]] else "page"
                )
                datacord = (
                    prediction[0][0],
                    prediction[0][1],
                    prediction[0][2],
                    prediction[0][3],
                )
                subset = 8
                if prediction[1] in ["amountwords_data", "payee_data"]:
                    subset = 15
                elif prediction[1] in ["memo_data"]:
                    subset = 8
                elif prediction[1] in ["routingno_data"]:
                    subset = 20
                txt = ""
                text_cords = []
                for word in ocr_words:
                    cords = (word[0][0], word[0][1], word[0][2], word[0][3])
                    if prediction[1] in [
                        "routingno_data",
                        "amount_data",
                        "payoraccountno_data",
                    ]:
                        formula = (
                            (
                                (word[0][0] in range(datacord[0], datacord[2]))
                                or (datacord[0] in range(word[0][0], word[0][2]))
                            )
                            and (word[0][1] in range(datacord[1] - 5, datacord[3] + 5))
                            and (
                                (
                                    word[0][2]
                                    in range(datacord[0] - subset, datacord[2] + subset)
                                    or (
                                        datacord[2]
                                        in range(
                                            word[0][0] - subset, word[0][2] + subset
                                        )
                                    )
                                )
                            )
                            and (
                                (
                                    word[0][3]
                                    in range(datacord[1] - subset, datacord[3] + subset)
                                )
                            )
                        )
                    else:
                        formula = (
                            (
                                (
                                    word[0][0]
                                    in range(datacord[0] - subset, datacord[2] + subset)
                                )
                            )
                            and (
                                word[0][1]
                                in range(datacord[1] - subset, datacord[3] + subset)
                            )
                            and (
                                word[0][2]
                                in range(datacord[0] - subset, datacord[2] + subset)
                            )
                            and (
                                (
                                    word[0][3]
                                    in range(datacord[1] - subset, datacord[3] + subset)
                                )
                            )
                        )
                    if formula:
                        ocr_score = word[2]
                        text_cords.append(cords)
                        txt += word[1] + " "
                if txt:
                    if ocr_score <= 60:
                        logic_cs = ocr_score
                    logic_cs += GLOBAL_MODEL_CONFIDENCE_SCORE_THRESHOLD["primary"]
                    min_word_x = min([word[0] for word in text_cords])
                    min_word_y = min([word[1] for word in text_cords])
                    max_word_x = max([word[2] for word in text_cords])
                    max_word_y = max([word[3] for word in text_cords])
                    cordinates = (min_word_x, min_word_y, max_word_x, max_word_y)
                    data_fetch.setdefault(prediction[1], []).append(
                        [
                            txt.strip(),
                            cordinates,
                            logic_cs,
                            page_no,
                            zone_type,
                            ocr_score,
                        ]
                    )
        return data_fetch

    def post_processing(self, deriving_txt, ocr_words, page_document):
        logical_data = {}
        for attribute_name in deriving_txt:
            if attribute_name in deriving_txt and deriving_txt[attribute_name]:
                if attribute_name in ["amount_data"]:
                    amount_data = re.findall(
                        "[\$\€\¥\₹\£\¥\₽\*]\s*[0-9]+[\.\,\s]*[0-9]*[\.\,\s]*[0-9]*[\.\,\s]*[0-9]*[\.\,\s]*[0-9]*|[0-9]+[\.]+[00]+",  # noqa
                        deriving_txt[attribute_name][0][0],
                    )
                    if amount_data:
                        deriving_txt[attribute_name][0][0] = amount_data[0].replace(
                            "*", ""
                        )

                elif attribute_name in ["routingno_data", "payoraccountno_data"]:
                    if deriving_txt[attribute_name][0][0] != []:
                        routing_data = re.findall(
                            r"[\d]+", deriving_txt[attribute_name][0][0]
                        )
                        if routing_data:
                            deriving_txt[attribute_name][0][0] = routing_data[0]
                        else:
                            deriving_txt[attribute_name][0][0] = ""

                elif attribute_name in ["checkdate_data"]:
                    data = deriving_txt[attribute_name][0][0]
                    regex_rule = r"\b(?:\d{4}[-.]\d{2}[-.]\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{2}|\d{2}-\d{2}-\d{2})\b|[\d]+[\.]+[\d]+[\.]+[\d]+|[\d]{2}\s+[\d]{2}\s+[\d]{4}|[\d]{6,8}"  # noqa
                    regex_patt = re.compile(regex_rule)
                    matched = re.search(regex_patt, data)
                    if matched:
                        value = matched.group(0).strip()
                        deriving_txt[attribute_name][0][0] = value
                    else:
                        date_finder = search_dates(data)
                        txt_data = ""
                        if date_finder:
                            for date_val in date_finder:
                                txt_data += date_val[0] + " "
                                break
                            deriving_txt[attribute_name][0][0] = txt_data
                        else:
                            deriving_txt[attribute_name][0][0] = ""

                elif attribute_name in ["payee_data"]:
                    data = deriving_txt[attribute_name][0][0]
                    payeee_data = deriving_txt["payee_data"][0][0]
                    number_words = [
                        "One",
                        "Two",
                        "Three",
                        "Four",
                        "Five",
                        "Seven ",
                        "Six",
                        "Eight",
                        "Nine",
                        "Ten",
                        "Eleven",
                        "Thirteen",
                        "Fourteen",
                        "Fifteen",
                        "Sixteen",
                        "Seventeen",
                        "Eighteen",
                        "Nineteen",
                        "Twenty",
                        "Thirty",
                        "Forty",
                        "Fifty",
                        "Sixty",
                        "Seventy",
                        "Eighty",
                        "Ninety",
                        "Hundred",
                        "Thousand",
                        "Million",
                        "Billion",
                        "Trillion",
                        "and",
                        "Cents",
                        "No Cents",
                        "$",
                        "*",
                        "Date",
                        "DATE",
                        "Hundr ed",
                        "Dollars",
                        "Fifty-Nine Eighty-Two",
                    ]
                    payee_list = payeee_data.split()
                    relevant_words = [
                        word for word in payee_list if word not in number_words
                    ]
                    data = " ".join(relevant_words)
                    deriving_txt[attribute_name][0][0] = data

                elif attribute_name in ["checkno_data"]:
                    data = deriving_txt[attribute_name][0][0]
                    checknum_data = deriving_txt["checkno_data"][0][0]
                    Check_no = re.search(r"\d+", checknum_data)
                    if Check_no:
                        matched = Check_no.group(0)
                        deriving_txt[attribute_name][0][0] = matched
                    else:
                        deriving_txt[attribute_name][0][0] = ""

        if (
            "checkdate_data" not in deriving_txt
            or deriving_txt["checkdate_data"][0][0] == ""
        ):
            labels = [
                {"label": "DDMMYYYY", "direction": "top"},
                {"label": "Date", "direction": "right"},
                {"label": "Date", "direction": "bottom"},
            ]
            regex = [
                "[0-9]{1,4}[\\.]+[0-9]{1,2}[\\.]+[0-9]{1,4}|[0-9]{1,2}[\\.]+[0-9]{1,2}[\\.]+[0-9]{1,4}|[0-9]{1,2}[\\/\\-]+[0-9]{1,2}[\\/\\-]+[0-9]{1,4}|[A-Za-z]{3,}[\\s\\.]+[0-9]{1,2}[\\,\\s\\.]+[0-9]{1,4}|[0-9]{2}[0-9]{2}[0-9]{4}"  # noqa
            ]
            data, radius_score, radius_cords = self.match_date_data(
                labels, regex, ocr_words, page_document
            )
            if data:
                logical_data.update(
                    {
                        "checkdate_data": [
                            [data, radius_cords, 50, 0, "page", radius_score]
                        ]
                    }
                )

        elif "payoraccountno_data" not in deriving_txt:
            data = self.match_acc_routing_data(ocr_words)
            data = re.findall(r"[\d]+", data.get("payoraccountno_data"))
            if data:
                logical_data.update(
                    {"payoraccountno_data": [[data, (0, 0, 0, 0), 50, 0, "page", 98.0]]}
                )

        elif "routingno_data" not in deriving_txt:
            data = self.value(ocr_words)
            data = re.findall(r"[\d]+", data.get("routingno_data"))
            if data:
                logical_data.update(
                    {"payoraccountno_data": [[data, (0, 0, 0, 0), 50, 0, "page", 98.0]]}
                )

        deriving_txt.update(logical_data)
        return deriving_txt

    def data_conversion_zone_level(self, model_prediction):
        new_dic = {}
        for i in model_prediction:
            new_dic.update({i[1]: []})
        for i in model_prediction:
            new_dic.setdefault(i[1], []).append(i[0])
        return new_dic

    def model_load(self, model_path):
        class_names = [
            "AmountWords_Data",
            "AmountWords_Label",
            "Amount_Data",
            "Amount_Label",
            "BankAddress_Data",
            "BankName_Data",
            "CheckDate_Data",
            "CheckDate_Label",
            "CheckNo_Data",
            "CheckNo_Label",
            "Memo_Data",
            "Memo_Label",
            "Payee_Data",
            "Payee_DataPayee_Data",
            "Payee_Label",
            "PayorAccountNo_Data",
            "PayorAccountNo_Label",
            "PayorAddress_Data",
            "PayorName_Data",
            "RoutingNo_Data",
            "RoutingNo_Label",
            "Singature_Data",
            "Singature_Label",
        ]
        session = Yolov5Onnx(
            classes=class_names,
            backend="onnx",
            weight=model_path,
            device="cpu",
        )

        return session

    def process_document(self):
        model_path = self.global_model()
        if model_path:
            try:
                model = self.model_load(model_path)
                all_attribute_data = {}
                attribute_log_data = copy.deepcopy(LOG_BOLIERPLATE)
                key_item_scores = dict.fromkeys(self.key_attributes.keys(), 0)
                all_data = {}
                for page_idx, page_document in enumerate(self.document_pages):
                    page_no = page_idx + 1
                    self.raw_texts = {}
                    self.raw_texts["page"] = self.ocr_output_by_page[page_idx]
                    ocr_words = self.raw_texts["page"]["words"]
                    model_prediction = model_pred(page_document, model)

                    model_prediction.sort(key=lambda x: x[2], reverse=True)
                    zone_data_conversion = self.data_conversion_zone_level(
                        model_prediction
                    )
                    Zone_level_prediction = filter_image_by_zone(
                        self.key_attributes, zone_data_conversion, page_document
                    )
                    deriving_txt = self.detect_text(
                        model_prediction, ocr_words, Zone_level_prediction, page_no
                    )
                    post_processed_data = self.post_processing(
                        deriving_txt, ocr_words, page_document
                    )
                    all_data.setdefault(page_no, []).append(post_processed_data)
                Key_item_value = self.mapping(all_data)
                attribute_log = Key_item_value
                image = cv2.imread(page_document)
                height, width = image.shape[:2]

                for key, results in Key_item_value.items():
                    for result in results:
                        if result[1] > key_item_scores[key]:
                            matched_cords = result[2]
                            matched_data = result[0].strip()
                            self.doc_log.update({key: attribute_log[key]})
                            key_item_scores[key] = result[1]
                            self.key_items[key] = result[0]
                            self.key_items_cs[key] = key_item_scores[key]
                            zone = result[3]
                            regex = result[4]
                            page_no = result[5]
                            attribute_log_data = copy.deepcopy(LOG_BOLIERPLATE)
                            attribute_log_data = update_log_data(
                                attribute_log_data,
                                page_words=None,
                                matched_cords=matched_cords,
                                matched_label=key,
                                label_cords=(0, 0, 0, 0),
                                value_text=matched_data,
                                attribute_type="string",
                                regex_value=regex,
                                approach="LayoutLM",
                            )
                            attribute_log_data.update({"zoning": zone})
                            attribute_log_data.update({"page_no": page_no})
                            attribute_log_data.update({"width": width})
                            attribute_log_data.update({"height": height})
                            all_attribute_data.update({key: attribute_log_data})
                doc_name = os.path.basename(self.document)
                generate_log_json(page_document, doc_name, all_attribute_data)

                if self.line_attributes:
                    table_processing = TableProcessing(
                        self.document,
                        self.document_pages,
                        self.line_attributes,
                        self.batch_process,
                        self.pg_num_doc_ids,
                        self.ocr_output_by_page,
                        self.ocr_engine,
                    )
                    self.line_items = table_processing.extract_line_attributes()

            except Exception as e:
                print("Process Document :: Exception :: ", e)
        else:
            print("NO MODEL FOUND")
