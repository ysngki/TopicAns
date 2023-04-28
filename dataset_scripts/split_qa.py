# %%
import xml.sax
import os
import re
import csv
import pandas as pd


# %%
# process xml file with this handler
class PostHandler(xml.sax.ContentHandler):
	def __init__(self, question_output_file="./generated_file/q_temp_output.csv",
				 answer_output_file="./generated_file/a_temp_output.csv"):
		self.CurrentData = ""

		self.q_f = open(question_output_file, "w", newline='')
		self.q_csv_writer = csv.writer(self.q_f)
		q_csv_head = ["Id", "PostTypeId", "Score", "AnswerCount", "AcceptedAnswerId", "Title", "Body", "Tags"]
		self.q_csv_writer.writerow(q_csv_head)

		self.a_f = open(answer_output_file, "w", newline='')
		self.a_csv_writer = csv.writer(self.a_f)
		a_csv_head = ["Id", "PostTypeId", "Score", "ParentId", "Body"]
		self.a_csv_writer.writerow(a_csv_head)

	def startElement(self, tag, attributes):
		self.CurrentData = tag
		if tag == "row":
			write_info = "*HonokaInfo"
			# get needed attributes
			Id = attributes["Id"]
			PostTypeId = attributes["PostTypeId"]
			Score = attributes["Score"]
			Body = attributes["Body"]

			# is question
			if PostTypeId == "1":
				AnswerCount = attributes["AnswerCount"]
				Tags = attributes["Tags"]
				Title = attributes["Title"]

				write_info += " " + Id + " " + PostTypeId + " " + Score + " " + AnswerCount

				AcceptedAnswerId = None
				try:
					AcceptedAnswerId = attributes["AcceptedAnswerId"]
				except:
					pass

				self.q_csv_writer.writerow([Id, PostTypeId, Score, AnswerCount, AcceptedAnswerId, Title, Body, Tags])

			# is answer
			elif PostTypeId == "2":
				ParentId = attributes["ParentId"]

				self.a_csv_writer.writerow([Id, PostTypeId, Score, ParentId, Body])

	def endElement(self, tag):
		self.CurrentData = ""

	def characters(self, content):
		pass

	# process special tokens
	def processStrangeCharacter(self, string):
		new_string = string.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("&apos;",
																									"'").replace(
			"&quot;", "\"")
		new_string = re.sub(r"<[^>]*>", "", new_string)
		return new_string


file_name = "Posts.xml"
# create handler
Handler = PostHandler()

with open(file_name, "r", encoding='utf-8') as f:
	i = 0
	# parse this line
	for line in f:
		# create XMLReader
		parser = xml.sax.make_parser()
		# turn off namespaces
		parser.setFeature(xml.sax.handler.feature_namespaces, 0)

		parser.setContentHandler(Handler)
		try:
			parser.feed(line)
			i += 1
		except:
			print(line)
			pass

		if i % 50000 == 0:
			print(f"\r{i}/51296934 = {i / 51296934}%", end="")

Handler.q_f.close()
Handler.a_f.close()
print()

# %%
# key: q_id, value: index
question_dic = {}
# index -> info
question_infos = []
now_index = 0

# find all question
with open("./generated_file/q_temp_output.csv", "r") as q_f:
	reader = csv.DictReader(q_f)
	for row in reader:
		q_id = row["Id"]
		temp = [q_id]

		question_dic[q_id] = now_index
		now_index += 1

		AcceptedAnswerId = row["AcceptedAnswerId"]

		if AcceptedAnswerId == "":
			f_best_answer = "-1"
		else:
			f_best_answer = AcceptedAnswerId

		temp.append(f_best_answer)
		temp.append(row["AnswerCount"])

		# [q_id, best_answer, answer_count]
		question_infos.append(temp)

question_number = now_index
print(question_number)

answers = []
for i in range(question_number):
	answers.append([])

with open("./generated_file/a_temp_output.csv", "r") as a_f:
	reader = csv.DictReader(a_f)
	for row in reader:
		q_id = row["ParentId"]
		# a_id, score
		temp = [row['Id'], row["Score"]]

		if question_dic.get(q_id) is None:
			continue
		answers[question_dic[q_id]].append(temp)

# aggregate
f_out = open("generated_file/question_info", "w")

for i in range(question_number):
	if question_infos[i][1] != "-1":
		# q_id + 1 + answer_count + best_answer_id + answers ...
		write_info = question_infos[i][0] + " 1 " + question_infos[i][2] + " " + question_infos[i][1]
		for item in answers[i]:
			if item[0] == question_infos[i][1]:
				continue
			write_info += " " + item[0]
		write_info += "\n"
	else:
		continue

	f_out.write(write_info)

f_out.close()
