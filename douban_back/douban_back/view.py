from django.http import HttpResponse
import json
import mysql.connector
import time

config = {
    'user': 'root',
    'password': 'abc123',
    'host': '127.0.0.1',
    'database': 'douban',
    'raise_on_warnings': True
}


def get_comment_list(request):
    movie_id = request.GET.get('movie_id')
    limit = request.GET.get('limit')
    offset = request.GET.get('offset')
    comment_list = []
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    sql = "select Username, Comment, Star from comment where MovieID = %s limit %s offset %s" % (movie_id, limit, offset)
    cursor.execute(sql)
    for item in cursor:
        comment_list.append({
            "username": item[0],
            "comment": item[1],
            "star": item[2]
        })
    cnx.close()
    # with conn as cursor:
    #     cursor.execute('select ID from movie')
    #     for i in range(cursor.rowcount):
    #         temp = cursor.fetchone()
    #         comment_list.append(temp)
    return HttpResponse(json.dumps(comment_list))


def get_score(request):
    comment = request.GET.get('comment')
    print(comment)
    time.sleep(1)
    return HttpResponse("2.8")
