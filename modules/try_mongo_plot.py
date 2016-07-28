#coding: utf-8

'''
db.col.find('answer_num' : {$type: 2}).forEach(
    function(x){
        x.answer_num = parseInt(x.answer_num);
        db.col.save(x);}
)
'''

import matplotlib.pyplot as plt
from mongoengine import connect, Document, StringField

connect('test')
class Life_q1(Document):
	follow_num = StringField(db_field='follow_num')
	visits_count = StringField(db_field='visits_count')
	answer_num = StringField(db_field='answer_num')
	title = StringField(db_field='title')

class Col(Document):
	answer_num = StringField(db_field='answer_num')
	title = StringField(db_field='title')

col = Col.objects[0]
print type(col.answer_num)


def get_focus():
	qs = Life_q1.objects(follow_num__gt=1500) # maybe follow_num is not interge
	print 'hello'
	for q in qs:
		print q.title


def show_scatter():
	x = []  # can use objects.only('follow_num')[:10000]
	y = []
	questions = Life_q1.objects[:10000]  
	for question in questions:
		x.append(question.follow_num)
		y.append(question.visits_count)

	plt.scatter(x,y)
	plt.show()