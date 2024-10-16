import json
from argparse import ArgumentParser
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import math

# https://github.com/Yuliang-Liu/MultimodalOCR/blob/main/scripts/LLaVA1_5.py

def _get_args():
	parser = ArgumentParser()
	parser.add_argument("--src", type=str, default="./OCRBench_Images")
	args = parser.parse_args()
	return args


OCRBench_score = {"Regular Text Recognition": 0, "Irregular Text Recognition": 0, "Artistic Text Recognition": 0,
                  "Handwriting Recognition": 0,
                  "Digit String Recognition": 0, "Non-Semantic Text Recognition": 0, "Scene Text-centric VQA": 0,
                  "Doc-oriented VQA": 0, "Doc-oriented VQA": 0,
                  "Key Information Extraction": 0, "Handwritten Mathematical Expression Recognition": 0}
AllDataset_score = {"IIIT5K": 0, "svt": 0, "IC13_857": 0, "IC15_1811": 0, "svtp": 0, "ct80": 0, "cocotext": 0, "ctw": 0,
                    "totaltext": 0, "HOST": 0, "WOST": 0, "WordArt": 0, "IAM": 0, "ReCTS": 0, "ORAND": 0,
                    "NonSemanticText": 0, "SemanticText": 0,
                    "STVQA": 0, "textVQA": 0, "ocrVQA": 0, "ESTVQA": 0, "ESTVQA_cn": 0, "docVQA": 0,
                    "infographicVQA": 0, "ChartQA": 0, "ChartQA_Human": 0, "FUNSD": 0, "SROIE": 0, "POIE": 0,
                    "HME100k": 0}
num_all = {"IIIT5K": 0, "svt": 0, "IC13_857": 0, "IC15_1811": 0, "svtp": 0, "ct80": 0, "cocotext": 0, "ctw": 0,
           "totaltext": 0, "HOST": 0, "WOST": 0, "WordArt": 0, "IAM": 0, "ReCTS": 0, "ORAND": 0, "NonSemanticText": 0,
           "SemanticText": 0,
           "STVQA": 0, "textVQA": 0, "ocrVQA": 0, "ESTVQA": 0, "ESTVQA_cn": 0, "docVQA": 0, "infographicVQA": 0,
           "ChartQA": 0, "ChartQA_Human": 0, "FUNSD": 0, "SROIE": 0, "POIE": 0, "HME100k": 0}

if __name__ == "__main__":
	args = _get_args()
	
	data = []
	with open(args.src, 'r') as jsonl_file:
		for line in jsonl_file:
			# 解析每一行的 JSON 数据
			json_obj = json.loads(line.strip())
			data.append(json_obj)
	
	for i in range(len(data)):
		data_type = data[i]["type"]
		dataset_name = data[i]["dataset_name"]
		answers = data[i]["answers"]
		if data[i].get('text', 0) == 0:
			continue
		predict = data[i]['text']
		data[i]['result'] = 0
		if dataset_name == "HME100k":
			if type(answers) == list:
				for j in range(len(answers)):
					answer = answers[j].strip().replace("\n", " ").replace(" ", "")
					predict = predict.strip().replace("\n", " ").replace(" ", "")
					if answer in predict:
						data[i]['result'] = 1
			else:
				answers = answers.strip().replace("\n", " ").replace(" ", "")
				predict = predict.strip().replace("\n", " ").replace(" ", "")
				if answers in predict:
					data[i]['result'] = 1
		else:
			if type(answers) == list:
				for j in range(len(answers)):
					answer = answers[j].lower().strip().replace("\n", " ")
					predict = predict.lower().strip().replace("\n", " ")
					if answer in predict:
						data[i]['result'] = 1
			else:
				answers = answers.lower().strip().replace("\n", " ")
				predict = predict.lower().strip().replace("\n", " ")
				if answers in predict:
					data[i]['result'] = 1
					
	if len(data) == 1000:
		for i in range(len(data)):
			if data[i].get("result", 100) == 100:
				continue
			OCRBench_score[data[i]['type']] += data[i]['result']
		recognition_score = OCRBench_score['Regular Text Recognition'] + OCRBench_score['Irregular Text Recognition'] + \
		                    OCRBench_score['Artistic Text Recognition'] + OCRBench_score['Handwriting Recognition'] + \
		                    OCRBench_score['Digit String Recognition'] + OCRBench_score['Non-Semantic Text Recognition']
		Final_score = recognition_score + OCRBench_score['Scene Text-centric VQA'] + OCRBench_score[
			'Doc-oriented VQA'] + OCRBench_score['Key Information Extraction'] + OCRBench_score[
			              'Handwritten Mathematical Expression Recognition']
		print("###########################OCRBench##############################")
		print(f"Text Recognition(Total 300):{recognition_score}")
		print("------------------Details of Recognition Score-------------------")
		print(f"Regular Text Recognition(Total 50): {OCRBench_score['Regular Text Recognition']}")
		print(f"Irregular Text Recognition(Total 50): {OCRBench_score['Irregular Text Recognition']}")
		print(f"Artistic Text Recognition(Total 50): {OCRBench_score['Artistic Text Recognition']}")
		print(f"Handwriting Recognition(Total 50): {OCRBench_score['Handwriting Recognition']}")
		print(f"Digit String Recognition(Total 50): {OCRBench_score['Digit String Recognition']}")
		print(f"Non-Semantic Text Recognition(Total 50): {OCRBench_score['Non-Semantic Text Recognition']}")
		print("----------------------------------------------------------------")
		print(f"Scene Text-centric VQA(Total 200): {OCRBench_score['Scene Text-centric VQA']}")
		print("----------------------------------------------------------------")
		print(f"Doc-oriented VQA(Total 200): {OCRBench_score['Doc-oriented VQA']}")
		print("----------------------------------------------------------------")
		print(f"Key Information Extraction(Total 200): {OCRBench_score['Key Information Extraction']}")
		print("----------------------------------------------------------------")
		print(
			f"Handwritten Mathematical Expression Recognition(Total 100): {OCRBench_score['Handwritten Mathematical Expression Recognition']}")
		print("----------------------Final Score-------------------------------")
		print(f"Final Score(Total 1000): {Final_score}")
	else:
		for i in range(len(data)):
			num_all[data[i]['dataset_name']] += 1
			if data[i].get("result", 100) == 100:
				continue
			AllDataset_score[data[i]['dataset_name']] += data[i]['result']
		for key in AllDataset_score.keys():
			print(f"{key}: {AllDataset_score[key] / float(num_all[key])}")
