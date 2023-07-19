import torch
from transformers import LayoutLMTokenizer
import os
import re
from common_global_model import (
    prediction,
    fetch_predicted_value,
    filter_image_by_zone,
    label_match,
    to_fetch_data,
    multi_line,
    currency,
    download_global_model,
    load_model_trained,
    mapping,
    user_configure_attribute,
    generate_log,
)
from table_processing import TableProcessing
from common_helper import generate_log_json
from data_extraction import *  # noqa

from common_helper import connect_aws_s3, connect_s3_resource

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")


class Purchase_Order:
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
        self.model_trained = ""

    def post_processing(
        self,
        data_fetch,
        page_no,
        zone_conf,
        Model_prediction,
        ocr_lines,
        ocr_words,
        multi_line_data,
    ):
        txt = ""
        cords = (0, 0, 0, 0)
        ocr_score = 10
        single_line_data = {}
        for data_find in data_fetch:
            single_line_data.update({data_find: []})

            if zone_conf[data_find]:
                zone_type = zone_conf[data_find][0]
            else:
                zone_type = "page"

            score = 0
            if len(data_fetch[data_find]) != 0:
                if len(data_fetch[data_find]) > 1:
                    remove_tmp = []
                    for val in data_fetch[data_find]:
                        value = val[0]
                        if value not in remove_tmp:
                            cords = data_fetch[data_find][0][1]
                            score = data_fetch[data_find][0][2]
                            ocr_score = data_fetch[data_find][0][3]
                            actual_label = data_fetch[data_find][0][4]
                            model_score = data_fetch[data_find][0][5]
                            txt += value + " "
                            remove_tmp.append(value)
                else:
                    txt = data_fetch[data_find][0][0]
                    cords = data_fetch[data_find][0][1]
                    score = data_fetch[data_find][0][2]
                    ocr_score = data_fetch[data_find][0][3]
                    actual_label = data_fetch[data_find][0][4]
                    model_score = data_fetch[data_find][0][5]

            spl_ch = "-=!@#$%^&*()_?><:}\\\/{[]}:."  # noqa
            if txt and data_find not in ["supplierphone_data", "faxno_data"]:
                for spl in spl_ch:
                    if spl in txt[0]:
                        txt = txt[1:]
            data = txt.strip()
            txt = ""

            if data_find in ["companyemail_data"] and data:
                data = data.replace(" ", "")
                regex_rule = r"\w+[@]\w+[\.]\w+[\.]\w+|\w+[@]\w+[\.]\w+|\w+[@]\w+[-]\w+[\.]\w+|\w+[\.]\w+[@]\w+[\.]\w+|\w+[-]\w+[@]\w+[-]\w+[\.]\w+|\w+[\.]\w+[\.]\w+[@]\w+[\.]\w+"
                regex_patt = re.compile(regex_rule)
                matched = re.search(regex_patt, data)
                if matched:
                    value = matched.group(0).strip()
                    single_line_data.setdefault(data_find, []).append(
                        [
                            value,
                            cords,
                            page_no,
                            zone_type,
                            ocr_score,
                            score,
                            actual_label,
                            model_score,
                        ]
                    )
                else:
                    single_line_data.setdefault(data_find, []).append(
                        [
                            "",
                            (None, None, None, None),
                            page_no,
                            zone_type,
                            0,
                            0,
                            actual_label,
                            0,
                        ]
                    )
            elif data_find in ["website_data"] and data:
                regex_rule = r"\bhttps[:]+\/\/\w+[\.]\w+\.\w+|\bhttps[:]+\/\/\w+[\.]\w+[-]\w+\.\w+|\bhttps[:]+\/\/\w+\.\w+|\bhttp[:]+\/\/\w+[\.]\w+\.\w+|\bhttp[:]+\/\/\w+[\.]\w+[-]\w+[\.]\w+|\bhttp[:]+\/\/\w+\.\w+|\bwww\b[.]\w+[.]\w+[.]\w+|\bwww\b[.]\w+[.]\w+|\bwww\b[.]\w+-+\w+[.]\w+|\bWWW\b[.]\w+[.]\w+"
                regex_patt = re.compile(regex_rule)
                matched = re.search(regex_patt, data)
                if matched:
                    value = matched.group(0).strip()
                    single_line_data.setdefault(data_find, []).append(
                        [
                            value,
                            cords,
                            page_no,
                            zone_type,
                            ocr_score,
                            score,
                            actual_label,
                            model_score,
                        ]
                    )
                else:
                    single_line_data.setdefault(data_find, []).append(
                        [
                            "",
                            (None, None, None, None),
                            page_no,
                            zone_type,
                            0,
                            0,
                            actual_label,
                            0,
                        ]
                    )

            elif data_find in ["ponumber_data"] and data:
                regex_rule = r"[\w]*[-]*[\w]+[0-9\/]+[A-Za-z0-9\/_,.-]+"
                regex_patt = re.compile(regex_rule)
                matched = re.search(regex_patt, data)
                if matched:
                    value = matched.group(0).strip()
                    single_line_data.setdefault(data_find, []).append(
                        [
                            value,
                            cords,
                            page_no,
                            zone_type,
                            ocr_score,
                            score,
                            actual_label,
                            model_score,
                        ]
                    )
                else:
                    single_line_data.setdefault(data_find, []).append(
                        [
                            "",
                            (None, None, None, None),
                            page_no,
                            zone_type,
                            0,
                            0,
                            actual_label,
                            0,
                        ]
                    )

            elif data_find in ["companyphone_data", "companyfaxno_data"] and data:
                data = re.sub(r"[A-Za-z]+|[\:]", "", data)
                for spl in spl_ch:
                    if spl in data[0]:
                        data = data[1:]
                data = data.strip()
                single_line_data.setdefault(data_find, []).append(
                    [
                        data,
                        cords,
                        page_no,
                        zone_type,
                        ocr_score,
                        score,
                        actual_label,
                        model_score,
                    ]
                )

            elif (
                data_find in ["totalamt_data", "subtotal_data", "taxamount_data"]
                and data
            ):
                regex_rule = r"[0-9]+[\.\,][0-9]+[\.\,]?[0-9]+"
                regex_patt = re.compile(regex_rule)
                matched = re.search(regex_patt, data)
                if matched:
                    value = matched.group(0).strip()
                    single_line_data.setdefault(data_find, []).append(
                        [
                            value,
                            cords,
                            page_no,
                            zone_type,
                            ocr_score,
                            score,
                            actual_label,
                            model_score,
                        ]
                    )
                else:
                    single_line_data.setdefault(data_find, []).append(
                        [
                            "",
                            (None, None, None, None),
                            page_no,
                            zone_type,
                            0,
                            0,
                            actual_label,
                            0,
                        ]
                    )

            else:
                if data_fetch[data_find]:
                    single_line_data.setdefault(data_find, []).append(
                        [
                            data,
                            cords,
                            score,
                            page_no,
                            zone_type,
                            ocr_score,
                            actual_label,
                            model_score,
                        ]
                    )

        Multiline_data = multi_line(
            Model_prediction, ocr_lines, zone_conf, page_no, multi_line_data
        )
        Currency_data = currency(ocr_words, Multiline_data, page_no)
        single_line_data.update(Multiline_data)
        single_line_data.update(Currency_data)
        return single_line_data

    def process_document(self):
        model_name = "purchase_order_global_model.pt"
        label_name = "labels.txt"
        folder_name = "Purchase-order"
        model_path, label_path = download_global_model(
            model_name, label_name, folder_name, self.s3_client
        )
        if not self.model_trained:
            model, label_map = load_model_trained(model_path, label_path)
            self.model_trained = model
        if model_path:
            try:
                multi_line = [
                    "supplier_address_data",
                    "suppliername_data",
                    "billtoname_data",
                    "billtoaddress_data",
                    "shiptoname_data",
                    "shiptoaddress_data",
                    "paymentterms_data",
                    "bankdetails_data",
                    "termsandconditions_data",
                ]
                key_item_scores = dict.fromkeys(self.key_attributes.keys(), 0)
                fetched_data = {}
                for page_idx, page_document in enumerate(self.document_pages):
                    page_no = page_idx + 1
                    self.raw_texts = {}
                    self.raw_texts["page"] = self.ocr_output_by_page[page_idx]
                    ocr_words = self.raw_texts["page"]["words"]
                    ocr_lines = self.raw_texts["page"]["words_by_line"]
                    Model_prediction = prediction(
                        page_document, ocr_words, model, label_map
                    )
                    Model_prediction_by_lines = prediction(
                        page_document, ocr_lines, model, label_map
                    )
                    fetch_key_value = fetch_predicted_value(Model_prediction)
                    Zone_level_prediction = filter_image_by_zone(
                        self.key_attributes, fetch_key_value, page_document
                    )
                    matched_data = label_match(fetch_key_value, ocr_words, multi_line)
                    data_fetch = to_fetch_data(matched_data, ocr_words)
                    post_processed_data = self.post_processing(
                        data_fetch,
                        page_no,
                        Zone_level_prediction,
                        Model_prediction_by_lines,
                        ocr_lines,
                        ocr_words,
                        multi_line,
                    )

                    fetched_data.setdefault(page_no, []).append(post_processed_data)
                attribute_log = mapping(fetched_data, self.key_attributes)
                data = user_configure_attribute(
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
                    self.key_attributes,
                )
                if data:
                    attribute_log.update(data)
                all_attribute_data = generate_log(
                    attribute_log,
                    key_item_scores,
                    self.doc_log,
                    self.key_items,
                    self.key_items_cs,
                    page_document,
                )
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
                print(" Global Model  ::: Exception ::: ", e)
