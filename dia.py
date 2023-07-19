import cv2
import copy
import os
import re
from common_global_model import (
    filter_image_by_zone,
    page_no,
)
from table_processing import TableProcessing
from config import LOG_BOLIERPLATE, PROCESS_ASSETS_S3_BUCKET
from common_helper import update_log_data
from common_helper import generate_log_json
from common_helper import connect_aws_s3, connect_s3_resource
from config import (
    GLOBAL_MODEL_PATH,
    DRIVE_TMP_PATH,
    GLOBAL_MODEL_CONFIDENCE_SCORE_THRESHOLD,
)

from data_extraction import DocumentProcessing
from cvu.detector.yolov5 import Yolov5 as Yolov5Onnx
from fuzzywuzzy import fuzz
from dateutil import parser
from yolo_prediction import model_pred


class DIA:
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
            # GLOBAL_MODEL_PATH = 'common-assets/global-modal/extraction_models'
            # DRIVE_TMP_PATH = '/tmp'
            folder_model = os.path.join(GLOBAL_MODEL_PATH, "Dia")
            local_path = os.path.join(DRIVE_TMP_PATH, GLOBAL_MODEL_PATH, "Dia")
            file_names = ["Dia.onnx"]
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
            print(" Model Download ::: Exception ::: ", e)
        return False

    def directional_text_formation(
        self,
        page_document,
        ngrams,
        words,
        index,
        label_direction="right",
    ):
        words.sort(key=lambda x: x[0][1], reverse=False)
        radius_words = []
        matched_label = ngrams[index]
        radius_score = 0
        radius_cords = (None, None, None, None)
        try:
            formed_line = []
            for word in words:
                if (
                    word[0][0] > matched_label[0][2]
                    and word[0][2] > matched_label[0][2]
                    and word[0][1] < matched_label[0][3]
                    and word[0][3] > matched_label[0][1]
                ):
                    formed_line.append(word)
            formed_line.sort(key=lambda x: x[0], reverse=False)
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
        radius_dic = {}
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
                words = ocr_words
                words.sort(key=lambda x: x[0][1], reverse=False)
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
                        radius_dic.update(
                            {label: [radius_text, radius_score, radius_cords]}
                        )
        return radius_dic

    def mapping(self, fetched_data):
        key_values = {}
        cords = (0, 0, 0, 0)
        for attribute_name in self.key_attributes:
            data_name = self.key_attributes[attribute_name].get("model_attribute_id")
            default_value = self.key_attributes[attribute_name].get("attribute_default")
            page_num = None
            if self.key_attributes[attribute_name].get("attribute_page_num"):
                page_num = self.key_attributes[attribute_name].get("attribute_page_num")
            regex = False
            zone = self.key_attributes[attribute_name].get("attribute_zone")
            if self.key_attributes[attribute_name].get("attribute_regex") != "":
                regex = eval(self.key_attributes[attribute_name].get("attribute_regex"))

            if data_name:
                data_name = data_name.lower()
                valid_page = False
                if page_num:
                    page_number = page_no(page_num, fetched_data)
                    for pg_no in page_number:
                        if (
                            pg_no in fetched_data
                            and data_name in fetched_data[pg_no][0]
                            and fetched_data[pg_no][0][data_name]
                        ):
                            image_number = pg_no - 1
                            valid_page = True
                            zone_type = fetched_data[pg_no][0][data_name][0][4]
                            attribute_score = fetched_data[pg_no][0][data_name][0][5]
                            score = attribute_score
                            if zone_type == "Invalid":
                                score -= GLOBAL_MODEL_CONFIDENCE_SCORE_THRESHOLD[
                                    "attribute_zoning"
                                ]

                            value = fetched_data[pg_no][0][data_name][0][0]
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
                            else:
                                value = value
                            cords = fetched_data[pg_no][0][data_name][0][1]
                            score = score

                            if regex:
                                regex = regex[0]
                            else:
                                regex = False

                            key_values.update(
                                {
                                    attribute_name: [
                                        (
                                            value.strip(),
                                            score,
                                            cords,
                                            zone,
                                            regex,
                                            image_number,
                                        )
                                    ]
                                }
                            )

                if valid_page == False:  # noqa
                    flag = False
                    for pg_no in fetched_data:
                        if (
                            data_name in fetched_data[pg_no][0]
                            and fetched_data[pg_no][0][data_name]
                            and fetched_data[pg_no][0][data_name][0][0] != ""
                        ):
                            image_number = pg_no - 1
                            flag = True
                            zone_type = fetched_data[pg_no][0][data_name][0][4]
                            attribute_score = fetched_data[pg_no][0][data_name][0][5]
                            score = attribute_score
                            if zone_type == "Invalid":
                                score -= GLOBAL_MODEL_CONFIDENCE_SCORE_THRESHOLD[
                                    "attribute_zoning"
                                ]

                            if page_num:
                                score -= GLOBAL_MODEL_CONFIDENCE_SCORE_THRESHOLD["page"]

                            value = fetched_data[pg_no][0][data_name][0][0]
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
                            cords = fetched_data[pg_no][0][data_name][0][1]
                            score = score
                            if regex:
                                regex = regex[0]
                            else:
                                regex = False
                            key_values.update(
                                {
                                    attribute_name: [
                                        (
                                            value.strip(),
                                            score,
                                            cords,
                                            zone,
                                            regex,
                                            image_number,
                                        )
                                    ]
                                }
                            )
                    else:
                        if flag == False:  # noqa
                            if default_value:
                                key_values.update(
                                    {
                                        attribute_name: [
                                            (
                                                default_value,
                                                100,
                                                (None, None, None, None),
                                                "page",
                                                False,
                                                0,
                                            )
                                        ]
                                    }
                                )
                            else:
                                if regex:
                                    regex = regex[0]
                                else:
                                    regex = False
                                key_values.update(
                                    {
                                        attribute_name: [
                                            (
                                                "",
                                                10,
                                                (None, None, None, None),
                                                zone,
                                                regex,
                                                0,
                                            )
                                        ]
                                    }
                                )
            else:
                key_val = DocumentProcessing()  # noqa
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
                key_items_output, attribute_log = key_val.process_document()
                pg_no = attribute_log[attribute_name]["page_no"]
                value = attribute_log[attribute_name]["value"]["text"]
                left, top, right, bottom = (
                    attribute_log[attribute_name]["value"]["left"],
                    attribute_log[attribute_name]["value"]["top"],
                    attribute_log[attribute_name]["value"]["right"],
                    attribute_log[attribute_name]["value"]["bottom"],
                )
                cords = (left, top, right, bottom)
                zone = attribute_log[attribute_name]["zoning"]
                regex = attribute_log[attribute_name]["regex"]
                score = key_items_output[attribute_name][0][1]
                if regex:
                    regex = regex[0]
                else:
                    regex = False
                key_values.update(
                    {attribute_name: [(value, score, cords, zone, regex, pg_no)]}
                )
        return key_values

    def detect_text(self, model_prediction, ocr_words, ocr_lines, zone_conf, page_no):
        data_fetch = {}
        model_prediction.sort(key=lambda x: x[0][1], reverse=False)
        for prediction in model_prediction:
            if "label" not in prediction[1] and prediction[1] and prediction[1]:
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
                words = ocr_words
                if prediction[1] in ["company_name_and_dba_name_data"]:
                    subset = 30
                    subset_y = 10
                elif prediction[1] in ["insurer_name_data"]:
                    words = ocr_lines
                elif prediction[1] in ["effective_date_data", "end_date_data"]:
                    subset = 18
                    subset_y = 0
                    ocr_words.sort(key=lambda x: x[0][1], reverse=False)

                txt = ""
                text_cords = []
                for word in words:
                    cords = (word[0][0], word[0][1], word[0][2], word[0][3])
                    if prediction[1] in [
                        "company_name_and_dba_name_data",
                        "effective_date_data",
                        "end_date_data",
                    ]:
                        formula = (
                            (
                                (word[0][0] in range(datacord[0] - subset, datacord[2]))
                                or (datacord[0] in range(word[0][0], word[0][2]))
                            )
                            and (
                                word[0][1]
                                in range(datacord[1] - subset_y, datacord[3] + 5)
                            )
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
                    elif prediction[1] in [
                        "insurer_name_data",
                    ]:
                        formula = (
                            (
                                (word[0][0] in range(datacord[0] - subset, datacord[2]))
                                or (
                                    datacord[0]
                                    in range(word[0][0] - subset, word[0][2])
                                )
                            )
                            and (word[0][1] in range(datacord[1] - subset, datacord[3]))
                            and (
                                word[0][3]
                                in range(datacord[1] - subset, datacord[3] + subset)
                            )
                            and (
                                (word[0][2] in range(datacord[0] - subset, datacord[2]))
                                or (
                                    datacord[2]
                                    in range(word[0][0] - subset, word[0][2])
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

    def post_processing(self, deriving_txt, ocr_words, page_document, pg_no):
        for attribute_name in deriving_txt:
            if attribute_name in deriving_txt and deriving_txt[attribute_name]:
                if attribute_name in ["company_name_and_dba_name_data"]:
                    txt = ""
                    text_cords = []
                    for prediction in deriving_txt[attribute_name]:
                        data_cords = deriving_txt[attribute_name][0][1]
                        data_conf = deriving_txt[attribute_name][0][2]
                        pg_no = deriving_txt[attribute_name][0][3]
                        pg_zone = deriving_txt[attribute_name][0][4]
                        ocr_conf = deriving_txt[attribute_name][0][5]
                        text_cords.append(data_cords)
                        txt += prediction[0] + " "
                    txt = txt.replace(",", ", ")
                    txt = re.sub(r"\s+", " ", txt)
                    if "name" in txt.lower():
                        idx = txt.lower().index("name")
                        txt = txt[: idx - 1].strip()
                    min_word_x = min([word[0] for word in text_cords])
                    min_word_y = min([word[1] for word in text_cords])
                    max_word_x = max([word[2] for word in text_cords])
                    max_word_y = max([word[3] for word in text_cords])
                    cordinates = (min_word_x, min_word_y, max_word_x, max_word_y)
                    data = [[txt, cordinates, data_conf, pg_no, pg_zone, ocr_conf]]
                    deriving_txt[attribute_name] = data

        if (
            "insurer_name_data" in deriving_txt
            and deriving_txt["insurer_name_data"][0][0] != ""
        ):
            data = deriving_txt["insurer_name_data"][0][0].lower().strip()
            if data.endswith("insurer"):
                data = data[:-7].strip()
            if data[-1] == ",":
                data = data[:-1].strip()

            deriving_txt["insurer_name_data"][0][0] = data

        if "effective_date_data" in deriving_txt:
            deriving_txt["effective_date_data"][0][0] = ""

        if (
            "effective_date_data" not in deriving_txt
            or deriving_txt["effective_date_data"][0][0] == ""
        ):
            labels = [
                {"label": "ISSUED, effective", "direction": "right"},
                {"label": "Renewed, effective", "direction": "right"},
            ]
            regex = [
                "[0-9]{1,4}[\\.]+[0-9]{1,2}[\\.]+[0-9]{1,4}|[0-9]{1,2}[\\.]+[0-9]{1,2}[\\.]+[0-9]{1,4}|[0-9]{1,2}[\\/\\-]+[0-9]{1,2}[\\/\\-]+[0-9]{1,4}|[A-Za-z]{3,}[\\s\\.]+[0-9]{1,2}[\\,\\s\\.]+[0-9]{1,4}|[0-9]{2}[0-9]{2}[0-9]{4}"  # noqa
            ]
            radius_dic = self.match_date_data(labels, regex, ocr_words, page_document)
            if radius_dic:
                label1 = False
                label2 = False
                if "ISSUED, effective" in radius_dic:
                    label1 = len(radius_dic.get("ISSUED, effective")[0])
                if "Renewed, effective" in radius_dic:
                    label2 = len(radius_dic.get("Renewed, effective")[0])
                if label1 and label2:
                    matched_label = (
                        "ISSUED, effective" if label1 > label2 else "Renewed, effective"
                    )
                    radius_text = radius_dic.get(matched_label)[0]
                    radius_score = radius_dic.get(matched_label)[1]
                    radius_cords = radius_dic.get(matched_label)[2]
                else:
                    radius_text = radius_dic.get(list(radius_dic)[0])[0]
                    radius_score = radius_dic.get(list(radius_dic)[0])[1]
                    radius_cords = radius_dic.get(list(radius_dic)[0])[2]

                if len(radius_text) <= 4:
                    radius_text = ""

                month_formats = [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December",
                ]

                for data in radius_text.split():
                    data = data.replace("/", "")
                    if data.isalpha() and data.title() not in month_formats:
                        radius_text = radius_text.replace(data, "")

                blacklist = ["at", "as", "#", "12:01", "12:01A"]
                idx = -1
                for txt in radius_text.split():
                    if txt in blacklist:
                        idx = radius_text.find(txt)
                        break
                    elif txt.startswith("#"):
                        idx = radius_text.find(txt)
                        break
                radius_text = radius_text[:idx].strip().replace(".", "")

                if "/ " in radius_text:
                    radius_text = radius_text.replace("/ ", "/")

                if radius_text.endswith("19"):
                    radius_text = radius_text[:-2].strip()

                pattern = re.findall(r".*19\s\d\d", radius_text)
                if pattern:
                    idx = radius_text.rfind(" ")
                    radius_text = radius_text[:idx] + radius_text[idx + 1:]

                if radius_text.endswith(","):
                    radius_text = radius_text[:-1].strip()

                if len(radius_text) > 4:
                    standard_format = "%m-%d-%Y"
                    try:
                        date_obj = parser.parse(radius_text)
                        radius_text = date_obj.strftime(standard_format)
                        if "2023" in radius_text:
                            radius_text = ""
                        elif radius_text.split("-")[-1][:2] == "20":
                            txt = radius_text.split("-")[-1].replace("20", "19")
                            txt_split = radius_text.split("-")
                            txt_split[-1] = txt
                            radius_text = "-".join(txt_split)
                    except ValueError:
                        radius_text = ""
                else:
                    radius_text = ""

                deriving_txt["effective_date_data"] = [
                    [radius_text, radius_cords, 60.0, pg_no, "page", radius_score]
                ]

        if (
            "end_date_data" in deriving_txt
            and deriving_txt["end_date_data"][0][0] != ""
        ):
            radius_text = deriving_txt["end_date_data"][0][0]
            month_formats = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]

            for data in radius_text.split():
                data = data.replace("/", "")
                if data.isalpha() and data.title() not in month_formats:
                    radius_text = radius_text.replace(data, "")

            if len(radius_text) <= 4:
                radius_text = ""

            blacklist = ["at", "as", "#", "12:01", "12:01A"]
            idx = -1
            for txt in radius_text.split():
                if txt in blacklist:
                    idx = radius_text.find(txt)
                    break
                elif txt.startswith("#"):
                    idx = radius_text.find(txt)
                    break
            radius_text = radius_text[:idx].strip().replace(".", "")

            if "/ " in radius_text:
                radius_text = radius_text.replace("/ ", "/")

            if radius_text.endswith("19"):
                radius_text = radius_text[:-2].strip()

            pattern = re.findall(r".*19\s\d\d", radius_text)
            if pattern:
                idx = radius_text.rfind(" ")
                radius_text = radius_text[:idx] + radius_text[idx + 1:]
            if radius_text.endswith(","):
                radius_text = radius_text[:-1].strip()

            if len(radius_text) > 4:
                standard_format = "%m-%d-%Y"
                try:
                    date_obj = parser.parse(radius_text)
                    radius_text = date_obj.strftime(standard_format)
                    if "2023" in radius_text:
                        radius_text = ""
                    elif radius_text.split("-")[-1][:2] == "20":
                        txt = radius_text.split("-")[-1].replace("20", "19")
                        txt_split = radius_text.split("-")
                        txt_split[-1] = txt
                        radius_text = "-".join(txt_split)
                except ValueError:
                    radius_text = ""
            else:
                radius_text = ""

            deriving_txt["end_date_data"][0][0] = radius_text

        return deriving_txt

    def data_conversion_zone_level(self, model_prediction):
        zone_data = {}
        for pred in model_prediction:
            zone_data.update({pred[1]: []})
        for pred in model_prediction:
            zone_data.setdefault(pred[1], []).append(pred[0])
        return zone_data

    def model_load(self, model_path):
        class_names = [
            "Company_Name_and_DBA_Name_Data",
            "Company_Name_and_DBA_Name_Label",
            "Effective_Date_Data",
            "Effective_Date_Label",
            "End_Date_Data",
            "End_Date_Label",
            "Insurer_Name_Data",
            "Insurer_Name_Label",
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
                    ocr_lines = self.raw_texts["page"]["words_by_line"]
                    model_prediction = model_pred(page_document, model)
                    zone_data_conversion = self.data_conversion_zone_level(
                        model_prediction
                    )
                    Zone_level_prediction = filter_image_by_zone(
                        self.key_attributes, zone_data_conversion, page_document
                    )
                    deriving_txt = self.detect_text(
                        model_prediction,
                        ocr_words,
                        ocr_lines,
                        Zone_level_prediction,
                        page_no,
                    )
                    post_processed_data = self.post_processing(
                        deriving_txt, ocr_words, page_document, page_no
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
