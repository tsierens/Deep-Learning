from flask import Flask, url_for
app = Flask(__name__)

@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        do_the_login()
    else:
        show_the_login_form()

@app.route('/')
def index():
    return 'Index Page'

@app.route('/hello.html')
def hello():
    return 'Hello World'


with app.test_request_context():
    print url_for('login')

@app.route('/user/todd/')
def profile(username): pass   


with app.test_request_context():
    print url_for('index')
    print url_for('login')
    print url_for('login', next='/')
    print url_for('profile', username='John Doe')

if __name__ == '__main__':
    app.run()