from flask import Flask, render_template, request, redirect, session
from database import MyDB
from dotenv import load_dotenv
import os
import pandas as pd 

#.env 로드
load_dotenv()

# Flask class 생성
# 생성자 함수에는 파일의 이름
app = Flask(__name__)

# 세션을 사용하기 위해 secret_key를 설정
app.secret_key = os.getenv('secret_key')

# DB server에 웹에서 사용할 table이 존재하는가?
# 존재한다면 아무 행동도 하지 않는다
# 존재 하지 않는다면 테이블을 생성
create_table = """
    CREATE TABLE 
    IF NOT EXISTS
    `user_list`
    (
        `id` varchar(32) primary key,
        `password` varchar(64) not null,
        `name` varchar(32)
    )
"""
# MyDB를 이용하여 서버의 접속하고 쿼리문 보낸다
web_db = MyDB()

# create_table 쿼리문을 실행
web_db.execute_query(create_table)

# 유저와 서버간의 데이터를 주고 받는 부분 생성
# api 생성

# localhost:5000/ 요청이 들어왔을때
@app.route('/')
def index():
    # 로그인 화면을 보여준다
    return render_template('signin.html')
    

# 로그인 화면에서 id password를 받아주는 주소를 생성
# /login, post 방식으로 데이터를 보낸다
@app.route('/login', methods=['post'])
def login():
    # post 방식으로 보낸 데이터는 request 안에 form에 존재(dict)
    # { 'user_id' : xxxx, 'user_pass' : xxxx }
    # 유저가 보낸 아이디값을 변수에 저장
    input_id = request.form['user_id']
    input_pass = request.form['user_pass']
    # 유저가 보낸 데이터를 확인
    print(f"[post] /login : {input_id}, {input_pass}")
    
    # 로그인의 로직
    # user_list table에서 유저가 보낸 id, password를 모두 존재하는 인덱스가 있는가?
    # sql 쿼리문 : select문
    login_query = """
        SELECT
        *
        FROM
        `user_list`
        WHERE `id` = %s AND `password` = %s
    """
    
    # execute_query()함수는 select문을 넣었을때 돌려주는 데이터의 타입은?
    # DataFrame => 데이터프레임의 길이가 0이면 => 로그인 실패
    # 길이가 1이라면 -> 로그인이 성공
    res_sql = web_db.execute_query(login_query, input_id, input_pass)
    if len(res_sql):
        # 로그인이 성공했다면 특정 페이지 보여준다 (render_template(파일의 이름))
        # 로그인이 성공했다면 특정 주소로 이동(redirect(주소값))
        # main.html와 같이 로그인을 한 유저의 이름을 보넨디
        # res_sql       id      password        name
        #             0 test      1234           kim
        # res_sql에서 이름만 추출한다면
        # res_sql.loc[0, 'name' ] // res_sql.iloc[0, 2]
        # res_sql['name'][0]
        logined_name = res_sql.loc[0, 'name']
        
        # return "로그인 성공"
        # session에 로그인 정보를 담는다
        # session의 데이터의 형태는 dict
        # session에 데이터를 대입입
        session['login_info'] = request.form
        # DataBase에 있는 sales records table을 로드
        select_query = """
            SELECT
            *
            FROM
            `sales records`
        """
        sales_data = web_db.execute_query(select_query)
        # sales_data의 타입은? DataFrame
        # DataFramedmf [ { }, { }, ... ] 형태로 변환
        list_data = sales_data.to_dict(orient='records')
        # html table 에 필요한 데이터를 컬럼의 이름들을 변수에 따로 저장
        # list_data에서 key
        cols = list_data[0].keys()
        
        # sales_data 
        # sales_data를 df에 copy()
        # -> 'Sales Channel' 컬럼을 기준으로 그룹화
        # 'Total Profit' 데이터를 그룹화 연산 합계
        df = sales_data.copy()
        group_df = df.groupby('Sales Channel')['Total Profit'].sum()
        # index의 값을 리스트로 생성
        index_list = list(group_df.index)
        # value의 값을 리스트로 생성 
        value_list = list(group_df.values)
        print(index_list)
        print(value_list)
        
        # render_template(파일의 이름, key = value, key2 = value2, ...)
        return render_template('main.html', name = logined_name, columns = cols, td_data = list_data, x_data = index_list, y_data = value_list)
    else:
        # 로그인이 실패했다면 로그인 페이지로 돌아간다
        # 로그인 주소로 이동('/')
        # return "로그인 실패"
        return redirect('/')
    

# 회원 가입 페이지를 보여주는 api 생성
@app.route("/signup")
def signup():
    # 로그인이 되어있는 상태라면 해당 페이지를 보여주지 않는다
    # 로그인이 되어있는 상태 : session에서 login_info 키가 존재하는가?
    # dict데이터에서 val1, val2, val3 = dict --> val, val2, val3에는 dict의 키 값들이 대입
    # dict 데이터에서 in 비교연산자 -> 키값에 존재 유무
    if 'login_info' in session:
        return render_template('home.html')
    else:
        return render_template('signup.html')

# 실제 회원 데이터를 받아서 DB에 저장하는 api 생성
@app.route('/signup2', methods=['post'])
def signup2():
    # form 태그를 이용해서 유저가 보낸 데이터 존재
    # id, password, name 의 값들을 각각 다른 변수에 저장
    id_ = request.form['user_id']
    password_ = request.form['user_pass']
    name_ = request.form['user_name']
    # id_, password_, name_ = request.for.values()
    
    # 변수를 확인 print
    print(f"[post[ /signup2 : {id_}, {password_}, {name_}")
    # insert query문 생성
    insert_query = """
        INSERT INTO `user_list` (`id`, `password`, `name`)
        VALUES (%s, %s, %s)
    """
    # try
    try:
        # MyDB class에 내장된 함수 execute_query() 함수를 호출
        web_db.execute_query(insert_query, id_, password_, name_, inplace=True)
        # 로그인 페이지로 이동
        return redirect('/')
    # except
    except Exception as e:
        print(e)
        # return을 회원가입이 실패한 경우
        # 회원가입 페이지로 이동
        return redirect('/signup')
    
@app.route('/graph')
def graph():
    # select 쿼리문을 이용하여 sales records 전체 데이터를 로드
    select_query = """
        SELECT
        *
        FROM
        `sales records`
    """
    # 결과 DataFrame에서
    df_ = web_db.execute_query(select_query)
    # Order Date컬럼의 데이터를 시계열 데이터로 변경하여 저장
    df_['Order Date'] = pd.to_datetime(df_['Order Date'])
    # 새로운 파생변수 Order Year을 생성하여 Order Date에서 4글자의 년도를 추출하여 저장
    df_['Order Year'] = df_['Order Date'].dt.year
    # Sales Channel과 Order Year를 기준으로 그룹화 Total Profit의 합계를 구한다
    group_df = df_.groupby(['Sales Channel', 'Order Year'])['Total Profit'].sum()
    # online 데이터를 따로 추출
    online_data = group_df.loc['Online']
    # offline 데이터를 따로 추출
    offline_data = group_df.loc['Offline']
    # online 데이터를 추출한 곳에서 index의 값을 리스트로 생성
    online_data_index = list(online_data.index)
    # online 데이터의 value를 리스트로 생성
    online_data_value = online_data.values.tolist()
    # offline 데이터의 value를 리스트로 생성
    offline_data_value = offline_data.values.tolist()
    # graph.html 파일에 online, offline 막대그래프 생성
    return render_template('graph.html', online_x = online_data_index, online_y = online_data_value, offline_y = offline_data_value)
# 웹서버의 실행
app.run(debug=True)