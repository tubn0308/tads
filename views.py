# -*- coding: utf-8 -*-
__author__ = "KPMG Consulting"
__copyright__ = "© 2020, KPMG Consulting Co"

from flask import Flask, render_template, request, Response, redirect, flash, url_for, send_from_directory
from flask_nav import Nav
from flask_nav.elements import Navbar, View
from flask_bootstrap import Bootstrap
import docx2txt
import os
import glob
import pathlib
import re
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta, timezone
import logging
import subprocess
import pandas as pd
from utils.log_util import LogUtil
from utils.keras_util import clear_session
from gensim.models import word2vec

log = LogUtil().log
LOGS_DIR = LogUtil().LOGS_DIR

from process.pattern_match.setup_util import Setup_util
from sub_process.contradiction_detection.predict import predict_contradiction
from process.svc_classification import SvcClassification
from process.similar_sentences import SimilerSentences
from process.contradiction_sentence import ContradictionSentence
from utils.extract_txt_from_pdf import Extract_txt_from_file
from process.pattern_match.ng_pattern_matching import Ng_pattern_matching
from process.pattern_match.ng_list_setup import Ng_list_setup
from utils import word2vec_util, config, file_util
from utils.contradiction_variabele import ContradictionVariabele
from process.create_sentence_data.seq2seq_predict import Predict
from process.create_sentence_data.merge_csv import merge_csv

import pdb

# Flask の起動
app = Flask(__name__, template_folder="app/templates", static_folder="app/static")

# ログ基本設定
# date_label = datetime.now(timezone(timedelta(hours=+9), 'JST')).strftime('%Y%m%d-%H%M%S')
# LOGS_DIR = 'logs/' + date_label
# os.makedirs(LOGS_DIR)

# logging.basicConfig(filename=LOGS_DIR + '/logger.log', level=logging.DEBUG)
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter('%(message)s')
# console.setFormatter(formatter)
# logger = logging.getLogger(__name__)

logging.info('Flask Start')
logging.info('log file: {}'.format(LOGS_DIR + '/flogging.log'))

rev = subprocess.check_output(['git', 'describe', '--always']).strip().decode('utf-8')
logging.info('Git rev: {}'.format(rev))

# configファイルの読み込み
conf = config.ConfigUtil()

# TODO パターンマッチと含意矛盾では異なるモデルを使う

# クラスのインスタンス化
# extract_txt_from_file = Extract_txt_from_file()
ng_pattern_matching = Ng_pattern_matching()
word2vec_util = word2vec_util.Word2VecUtil(config.get_property('word2vec', 'Word2VecFullModel'))
ng_list_setup = Ng_list_setup()
setup_util = Setup_util()
contradiction_var = ContradictionVariabele()
contradiction_sentence = ContradictionSentence()

# NavBarのセットアップとFlask用Boottrapのセットアップ
nav = Nav()
# NaviBarを作って適応
nav.init_app(app)
bootstrap = Bootstrap(app)

# ファイル読み込みの準備
UPLOAD_FOLDER = 'data/ui_uploads'
ALLOWED_EXTENSIONS = set(['pdf','docx','doc','xlsx','csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

# 変数
CONTRADICTION_MODEL_PATH = "./model/contradiction_sentence"
FEEDBACK_PATH = config.get_property('rte', 'FeedbackPath')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# ファイルをアップロードする関数
def upload_file(request_name):
    send_data = request.files[request_name]
    if send_data and allowed_file(send_data.filename):
        filename = secure_filename(send_data.filename)
        send_data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        logging.info('file upload succesess: {}'.format(filename))
        return filename


# NavBarで表示する内容を設定
@nav.navigation()
def navbar():
    return Navbar(
        'GRACE',
        View('Home', 'index'),
        View('Check Prohibited Expression', 'check_word'),
        View('Check Contradiction', 'check_contradiction')
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    error_code = [0,0,0,0] # 順に「ファイルアップロード」「類似単語抽出N数設定」「品詞設定」「保存名」のエラーフラグ
    error_message = []
    noun_pos_value = setup_util.noun_pos_value
    noun_pos_checked = setup_util.noun_pos_checked
    if request.method == 'GET':
        return render_template('index.html', error_code=error_code, error_message=error_message, noun_pos_value=noun_pos_value, noun_pos_checked=noun_pos_checked)
    if request.method == 'POST':
        # データ入力チェック
        # TODO エラーの際に、データを引き継ぐ設定にする
        if not request.files['send_data']:
            error_code[0] = 1
            message = config.get_property('message', 'FileUnselectedError')
            error_message.append(message)
        if not request.form["extract_num"]:
            error_code[1] = 1
            message = config.get_property('message', 'ExtractNumberUnselectedError')
            error_message.append(message)
        if not request.form.getlist("extract_pos"):
            error_code[2] = 1
            message = config.get_property('message', 'ExtractPosUnselectedError')
            error_message.append(message)
        if not request.form['ng_file_name']:
            error_code[3] = 1
            message = config.get_property('message', 'NgFileNameUninputError')
            error_message.append(message)
        logging.debug('similar ngword extraction')

        # ファイルアップロード
        if request.files['send_data']:
            filename = setup_util.upload_file(app.config['UPLOAD_FOLDER'], request.files['send_data'])
            file_path = UPLOAD_FOLDER + '/' + filename
            # ファイル内フォーマットチェック
            if ng_list_setup.format_check(file_path) == 0:
                error_code[0] = 1
                message = config.get_property('message', 'FileFormatError')
                error_message.append(message)

        # ファイルディレクトリ作成
        if request.form['ng_file_name']:
            output_dir_name = request.form['ng_file_name']
            if os.path.isdir(LOGS_DIR + output_dir_name):
                error_code[3] = 1
                message = config.get_property('message', 'NgFileNameDuplicationError')
                error_message.append(message)
            else:
                os.makedirs(LOGS_DIR + output_dir_name)

        # エラーがあれば、エラー表示
        if 1 in error_code:
            return render_template('index.html', error_code=error_code, error_message=error_message, noun_pos_value=noun_pos_value, noun_pos_checked=noun_pos_checked)

        # TODO NG類似表現抽出
        ng_list_setup.ng_similar_list_generator(file_path)

        return render_template('index.html', error_code=error_code, error_message=error_message, noun_pos_value=noun_pos_value, noun_pos_checked=noun_pos_checked)

@app.route('/ng_check_setup')
def ng_check_setup():
    error_code = [0,0,0,0] # 順に「ファイルアップロード」「類似単語抽出N数設定」「品詞設定」「保存名」のエラーフラグ
    error_message = []
    noun_pos_value = setup_util.noun_pos_value
    noun_pos_checked = setup_util.noun_pos_checked
    if request.method == 'GET':
        return render_template("ng_check_setup.html", error_code=error_code, error_message=error_message, noun_pos_value=noun_pos_value, noun_pos_checked=noun_pos_checked)

@app.route('/check_word', methods=['GET', 'POST'])
def check_word():
    if request.method == 'GET':
        return render_template('check_word.html')
    if request.method == 'POST':
        filename = upload_file('send_data')
        sentence = file_util.extract_contents(filename)
        result, test_check, ngwords_value, ngword_note_value, similarity_checkable_words_list = ng_pattern_matching.sentences_check(sentence, word2vec_util.model)
        # TODO データがuploadsフォルダに残り続けるので、削除が必要（ここで削除か別で作るかは検討）

    return render_template('check_word.html', text=zip(result, test_check, similarity_checkable_words_list), border_text=zip(result, test_check, ngwords_value, ngword_note_value, similarity_checkable_words_list))

@app.route('/check_contradiction', methods=['GET', 'POST'])
def check_contradiction():
    if request.method == 'GET':
        return render_template('check_contradiction.html')
    if request.method == 'POST':
        
        # フォームの文書をdfに変換
        if request.form.get('sourceform') == "document":
            source_name = upload_file('source')
            source_df_dict = file_util.extract_contents(source_name)
            if type(source_df_dict) is str:
                sentences_list = re.split("\r|\n|。", source_df_dict)
                sentences_list = [sentence for sentence in sentences_list if sentence != ""]
                # source_df_dict = file_util.sentences_file_df_dict(UPLOAD_FOLDER + '/' + source_name, sentences_list, "sentence2")
                source_df_dict = file_util.sentences_file_df_dict(source_name, sentences_list, "sentence2")
        else:
            sentence = request.form.get('source')
            sentences_list = re.split("\r|\n|。", sentence)
            sentences_list = [sentence for sentence in sentences_list if sentence != ""]
            source_df_dict = file_util.sentences_file_df_dict("source_document", sentences_list, "sentence2")
        if request.form.get('targetform') == "document":
            target_name = upload_file('target')
            target_df_dict = file_util.extract_contents(target_name)
            if type(target_df_dict) is str:
                sentences_list = re.split("\r|\n|。", target_df_dict)
                sentences_list = [sentence for sentence in sentences_list if sentence != ""]
                # target_df_dict = file_util.sentences_file_df_dict(UPLOAD_FOLDER + '/' + target_name, sentences_list, "sentence1")
                target_df_dict = file_util.sentences_file_df_dict(target_name, sentences_list, "sentence1")
        else:
            sentence = request.form.get('target')
            sentences_list = re.split("\r|\n|。", sentence)
            sentences_list = [sentence for sentence in sentences_list if sentence != ""]
            target_df_dict = file_util.sentences_file_df_dict("target_document", sentences_list, "sentence1")

        # 文章削減（オプションでスキップ）
        if request.form.get('reduce') == "reduce-on":
            log.info('svc_classification')
            svc = SvcClassification(word2vec_util)
            source_df_dict = svc.reduce(source_df_dict, 'sentence2')
            target_df_dict = svc.reduce(target_df_dict, 'sentence1')

        # 類似文章抽出
        log.info('similar_sentences')
        similar_sentence = SimilerSentences(word2vec_util)
        extract_rank = int(request.form.get('extract_rank'))
        if extract_rank == 0:
            extract_rank = 99999999
        similar_df = similar_sentence.detect_similar_sentences(source_df_dict, target_df_dict, extract_rank=extract_rank)

        # 矛盾予測
        # 単語予測のON,OFFにより単語矛盾予測のON,OFFを変更
        if request.form.get('word_check') == "word_check-on":
            contradiction_word = True
        else:
            contradiction_word = False
        log.info('contradiction_detection')
        result_df = predict_contradiction(similar_df, contradiction_word)

        # 詳細タブ用
        sentence1_list = result_df["origin_sentence1"].values.tolist()
        sentence2_list = result_df["origin_sentence2"].values.tolist()
        predict_list = result_df["predict_label"].values.tolist()
        correct_probability_list = result_df["predict_probability_○"].values.tolist()
        correct_probability_list = [round(num, 3) for num in correct_probability_list]
        contradiction_probability_list = result_df["predict_probability_×"].values.tolist()
        contradiction_probability_list = [round(num, 3) for num in contradiction_probability_list]
        # 単語矛盾予測のON,OFFによって、変数の内容を変更
        if request.form.get('word_check') == "word_check-on":
            word_predict1_list = result_df["word_predict1"].values.tolist()
            word_predict2_list = result_df["word_predict2"].values.tolist()
        else:
            word_predict1_list = []
            word_predict2_list = []
            predict_list_num = 0
            for sentence2, sentence1 in zip(sentence2_list, sentence1_list):
                predict_label = predict_list[predict_list_num]
                word_predict1_list.append([predict_label for _ in sentence1])
                word_predict2_list.append([predict_label for _ in sentence2])
                predict_list_num += 1
        
        # 概要タブ用
        sentence2_unique_list = []
        contradiction_for_sentence2_list = []
        contradiction_count_list = []
        sentence2_origin_list = ["".join(word_list) for word_list in sentence2_list]
        sentence1_list_for_modal = []
        sentence2_list_for_modal = []
        predict_list_for_modal = []
        correct_probability_list_for_modal = []
        contradiction_probability_list_for_modal =[]
        word_predict1_list_for_modal = []
        word_predict2_list_for_modal = []
        for sentence2 in list(dict.fromkeys(sentence2_origin_list)):
            sentence2_unique_list.append(sentence2)
            sentence2_index = [i for i, word_list in enumerate(sentence2_list) if "".join(word_list) == sentence2]
            contradiction_check = max([predict for i, predict in enumerate(predict_list) if i in sentence2_index])
            contradiction_for_sentence2_list.append(contradiction_check)
            contradiction_count = [predict for i, predict in enumerate(predict_list) if i in sentence2_index].count(1)
            contradiction_count_list.append(contradiction_count)
            sentence1_list_for_modal.append([x for i, x in enumerate(sentence1_list) if i in sentence2_index])
            sentence2_list_for_modal.append([x for i, x in enumerate(sentence2_list) if i in sentence2_index])
            predict_list_for_modal.append([x for i, x in enumerate(predict_list) if i in sentence2_index])
            correct_probability_list_for_modal.append([x for i, x in enumerate(correct_probability_list) if i in sentence2_index])
            contradiction_probability_list_for_modal.append([x for i, x in enumerate(contradiction_probability_list) if i in sentence2_index])
            word_predict1_list_for_modal.append([x for i, x in enumerate(word_predict1_list) if i in sentence2_index])
            word_predict2_list_for_modal.append([x for i, x in enumerate(word_predict2_list) if i in sentence2_index])
        
        clear_session()
        
        # 変数を格納
        contradiction_var.var_update(sentence1_list, sentence2_list, predict_list, correct_probability_list, contradiction_probability_list, word_predict1_list, word_predict2_list, sentence2_unique_list, contradiction_for_sentence2_list, contradiction_count_list, sentence1_list_for_modal, sentence2_list_for_modal, predict_list_for_modal, correct_probability_list_for_modal, contradiction_probability_list_for_modal, word_predict1_list_for_modal, word_predict2_list_for_modal, contradiction_word, similar_df)
    
    return render_template('check_contradiction.html',
        overview_text=zip(sentence2_unique_list, contradiction_for_sentence2_list, contradiction_count_list),
        overview_modal=zip(sentence1_list_for_modal, sentence2_list_for_modal, predict_list_for_modal, correct_probability_list_for_modal, contradiction_probability_list_for_modal, word_predict1_list_for_modal, word_predict2_list_for_modal),
        detail_text=zip(sentence1_list, sentence2_list, predict_list, correct_probability_list, contradiction_probability_list, word_predict1_list, word_predict2_list),
        detail_modal=zip(sentence1_list, sentence2_list, predict_list, correct_probability_list, contradiction_probability_list, word_predict1_list, word_predict2_list),
        contradiction_word=contradiction_word
    )

@app.route('/check_contradiction/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        # フィードバックの内容をcsvに保存
        fb_csv = pd.read_csv(FEEDBACK_PATH, header=0)
        fb_no = len(fb_csv["origin_No"])
        fb_text1 = request.form["text1"]
        fb_text2 = request.form["text2"]
        fb_category = request.form["category"]
        fb_item = request.form["item"]
        fb_label = request.form["label"]
        fb_target = "-"
        fb_create_flg = 0
        fb_df = pd.Series(
            [fb_no, fb_text1, fb_text2, fb_category, fb_item, fb_label, fb_target, fb_create_flg],
            index=["origin_No", "origin_t1", "source", "origin_category", "origin_item", "gold_label", "target", "create_flg"],
            name="append_line"
        )
        fb_csv = fb_csv.append(fb_df)
        fb_csv.to_csv(FEEDBACK_PATH, index=False)
        feedback = True

        # 変数を取得
        sentence1_list, sentence2_list, predict_list, correct_probability_list, contradiction_probability_list, word_predict1_list, word_predict2_list, sentence2_unique_list, contradiction_for_sentence2_list, contradiction_count_list, sentence1_list_for_modal, sentence2_list_for_modal, predict_list_for_modal, correct_probability_list_for_modal, contradiction_probability_list_for_modal, word_predict1_list_for_modal, word_predict2_list_for_modal, contradiction_word, _ = contradiction_var.var_call()

    return render_template('check_contradiction.html',
    overview_text=zip(sentence2_unique_list, contradiction_for_sentence2_list, contradiction_count_list),
    overview_modal=zip(sentence1_list_for_modal, sentence2_list_for_modal, predict_list_for_modal, correct_probability_list_for_modal, contradiction_probability_list_for_modal, word_predict1_list_for_modal, word_predict2_list_for_modal),
    detail_text=zip(sentence1_list, sentence2_list, predict_list, correct_probability_list, contradiction_probability_list, word_predict1_list, word_predict2_list),
    detail_modal=zip(sentence1_list, sentence2_list, predict_list, correct_probability_list, contradiction_probability_list, word_predict1_list, word_predict2_list),
    contradiction_word=contradiction_word,
    feedback=feedback)

@app.route('/check_contradiction/word_check')
def check_contradiction_word():
    # if request.method == 'POST':
    log.info("contradiction word check execute")
    # 変数を取得
    sentence1_list, sentence2_list, predict_list, correct_probability_list, contradiction_probability_list, word_predict1_list, word_predict2_list, sentence2_unique_list, contradiction_for_sentence2_list, contradiction_count_list, sentence1_list_for_modal, sentence2_list_for_modal, predict_list_for_modal, correct_probability_list_for_modal, contradiction_probability_list_for_modal, word_predict1_list_for_modal, word_predict2_list_for_modal, contradiction_word, similar_df = contradiction_var.var_call()

    # 予測再実行
    log.info('contradiction_detection')
    result_df = predict_contradiction(similar_df, True)

    # 変数更新
    contradiction_word = True
    word_predict1_list = result_df["word_predict1"].values.tolist()
    word_predict2_list = result_df["word_predict2"].values.tolist()
    sentence2_origin_list = ["".join(word_list) for word_list in sentence2_list]
    word_predict1_list_for_modal = []
    word_predict2_list_for_modal = []
    for sentence2 in list(dict.fromkeys(sentence2_origin_list)):
        sentence2_index = [i for i, word_list in enumerate(sentence2_list) if "".join(word_list) == sentence2]
        word_predict1_list_for_modal.append([x for i, x in enumerate(word_predict1_list) if i in sentence2_index])
        word_predict2_list_for_modal.append([x for i, x in enumerate(word_predict2_list) if i in sentence2_index])

    # 変数を格納
    contradiction_var.var_update(sentence1_list, sentence2_list, predict_list, correct_probability_list, contradiction_probability_list, word_predict1_list, word_predict2_list, sentence2_unique_list, contradiction_for_sentence2_list, contradiction_count_list, sentence1_list_for_modal, sentence2_list_for_modal, predict_list_for_modal, correct_probability_list_for_modal, contradiction_probability_list_for_modal, word_predict1_list_for_modal, word_predict2_list_for_modal, contradiction_word, similar_df)

    return render_template('check_contradiction.html',
        overview_text=zip(sentence2_unique_list, contradiction_for_sentence2_list, contradiction_count_list),
        overview_modal=zip(sentence1_list_for_modal, sentence2_list_for_modal, predict_list_for_modal, correct_probability_list_for_modal, contradiction_probability_list_for_modal, word_predict1_list_for_modal, word_predict2_list_for_modal),
        detail_text=zip(sentence1_list, sentence2_list, predict_list, correct_probability_list, contradiction_probability_list, word_predict1_list, word_predict2_list),
        detail_modal=zip(sentence1_list, sentence2_list, predict_list, correct_probability_list, contradiction_probability_list, word_predict1_list, word_predict2_list),
        contradiction_word=contradiction_word
    )

def extract_contradiction_model_info():
    model_path_list = [p for p in glob.glob(CONTRADICTION_MODEL_PATH + '/*.model')]
    model_list = [os.path.basename(p) for p in glob.glob(CONTRADICTION_MODEL_PATH + '/*.model')]
    dt_list = []
    size_list = []
    JST = timezone(timedelta(hours=+9), 'JST')
    for model_path in model_path_list:
        path_info = pathlib.Path(model_path)
        dt = datetime.fromtimestamp(path_info.stat().st_mtime, JST)
        dt_list.append(dt.strftime('%Y年%m月%d日 %H:%M:%S'))
        size = path_info.stat().st_size
        size_list.append(round(size/1024.0, 0))
    return model_list, dt_list, size_list

@app.route('/admin_index')
def admin_index():
    return render_template("admin_index.html")

@app.route('/contradiction_setup')
def contradiction_setup():
    model_list, dt_list, size_list = extract_contradiction_model_info()
    return render_template("contradiction_setup.html", model_data=zip(model_list, dt_list, size_list))

# フィードバックを元にモデルを再学習
@app.route('/contradiction_setup/retrain')
def contradiction_retrain():
    # 引数定義
    model_path = "./model/seq_gan" # モデルの読み込みディレクトリ
    # 実行
    df = pd.read_csv(FEEDBACK_PATH, header=0)
    df = df[df.create_flg == 0]
    label_dict = {"correct": "○", "false": "×"}
    for label in ["correct", "false"]:
        model_files = os.listdir(model_path+"/"+label)
        item_list = [f for f in model_files if os.path.isdir(os.path.join(model_path+"/"+label, f))]
        for item in item_list:
            if item not in df["origin_item"].values.tolist():
                continue
            if label_dict[label] not in df[df.origin_item == item].gold_label.values.tolist():
                continue
            label_df = df[df.gold_label == label_dict[label]]
            item_df = label_df[df.origin_item == item]
            df_len = len(item_df["source"])
            predict = Predict(label+"/"+item, model_path, df_len)
            end = False
            num = 0
            while not end:
                return_df = predict.exec(item_df, num, word2vec_util)
                item_df, end_flg = predict.repeat_exec(return_df)
                num += 1
                end = end_flg
    df = df.replace({'create_flg': {0: 1}})
    df.to_csv(FEEDBACK_PATH, index=False)
    clear_session()

    # outputを統合
    merge_df = merge_csv(config.get_property('rte', 'RetrainInterPath'))

    # モデル再学習
    contradiction_sentence.retrain(20, merge_df)
    clear_session()

    # UI用変数
    model_list, dt_list, size_list = extract_contradiction_model_info()
    retrain_success = True

    return render_template("contradiction_setup.html", model_data=zip(model_list, dt_list, size_list), retrain_success=retrain_success)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8888)