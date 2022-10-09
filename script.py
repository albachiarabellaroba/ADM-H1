#EX 1

#INTRODUCTION

#Say "Hello, World!" With Python
ciao='Hello, World!'
print(ciao)

#Python If-Else
n = int(input().strip())
if n % 2 != 0:
    print('Weird')
elif n % 2 == 0 and 2<=n<=5 or n % 2 == 0 and n>20:
    print('Not Weird')
elif n % 2 == 0 and 6<=n<=20:
    print('Weird')

#Arithmetic Operators
a = int(input())
b = int(input())
print(a+b)
print(a-b)
print(a*b)

#Python: Division
a = int(input())
b = int(input())
intdiv=a//b
fldiv=a/b
print(intdiv)
print(fldiv)

#Loops
n = int(input())
for i in range(n):
    print(i**2)

#Write a function
def is_leap(year):
    leap = False
    if year % 4 == 0 and year % 100 != 0:
        leap = True
    elif year % 400 == 0:
        leap = True
    return leap

#Print Function
n = int(input())
new=[]
for i in range(1,n+1):
    new.append(str(i))
string=''.join(new)
print(string)

#BASIC DATA TYPES

#Find the Runner-Up Score!   #wrong code
n = int(input())
arr = map(int, input().split())
A=list(arr)
first=max(A)
for i in A:
    if i==first:
        A.remove(i)
print(A)
print(max(A))

#List Comprehensions
x = int(input())
y = int(input())
z = int(input())
n = int(input())

combination=[[x1,y1,z1] for x1 in range(x+1) for y1 in range(y+1) for z1 in range(z+1)]
output=list(filter(lambda x:sum(x)!=n,combination))
print(output)

#Nested Lists
names = [''] * 2
scores = [float('inf')] * 2
for _ in range(int(input())):
    name = input()
    score = float(input())
    if score in scores:
        if score == scores[0]:
            names[0] = [names[0]] + [name]
        else:
            names[1] = [names[1]] + [name]
    elif score < scores[0]:
        scores[1], scores[0] = scores[0], score
        names[1], names[0] = names[0], name
    elif score < scores[1]:
        scores[1] = score
        names[1] = name
if isinstance(names[1], list):
    names[1].sort()
    print(*names[1], sep='\n')
else: print(names[1])

#Finding the percentage
n = int(input())
student_marks = {}
for _ in range(n):
    name, *line = input().split()
    scores = list(map(float, line))
    student_marks[name] = scores
query_name = input()
for i in student_marks:
    if i==query_name:
        average=(sum(student_marks[query_name]))/len(student_marks[query_name])
print("{:.2f}".format(average))

#Lists
N = int(input())
lista=[]
for i in range(N):
    n=input().split()
    if n[0]=='insert':
        lista.insert(int(n[1]),int(n[2]))
    elif n[0]=='print':
        print(lista)
    elif n[0]=='remove':
        if int(n[1]) in lista:
            lista.remove(int(n[1]))
    elif n[0]=='append':
        lista.append(int(n[1]))
    elif n[0]=='sort':
        lista.sort()
    elif n[0]=='pop':
        lista.pop()
    elif n[0]=='reverse':
        lista.reverse()

#Tuples
n=int(input())
t=input().split()
l=[]
for i in t:
    l.append(t)
tupla=tuple(l)
print(hash(tupla))

#STRINGS

#sWAP cASE
def swap_case(s):
    l=''
    for i in s:
        if i.islower()==True:
            l=l+i.upper()
        else:
            l=l+i.lower()
    return l

#String Split and Join
def split_and_join(line):
    line=line.split(' ')
    line='-'.join(line)
    return line

#What's Your Name?
def print_full_name(first, last):
    return print('Hello '+first+' '+last+'! You just delved into python.')

#Mutations
def mutate_string(string, position, character):
    mut=string[:position]+character+string[position+1:]
    return mut

#Find a string
def count_substring(string, sub_string):
    count=0
    num=0
    check=len(sub_string)
    for i in range(len(string)-(check)+1):
        for k in range(check):
            if string[k]==sub_string[k]:
                count=count+1
        if count==check:
            num=num+1
    return num

#String Validators
    s = input()
    alnum=0
    alpha=0
    digit=0
    lower=0
    upper=0
    for i in s:
        if i.isalnum()==True:
            alnum=alnum+1
        if i.isalpha()==True:
            alpha=alpha+1
        if i.isdigit()==True:
            digit=digit+1
        if i.islower()==True:
            lower=lower+1
        if i.isupper()==True:
            upper=upper+1
    if alnum>0:
        print(True)
    else:
        print(False)
    if alpha>0:
        print(True)
    else:
        print(False)
    if digit>0:
        print(True)
    else:
        print(False)
    if lower>0:
        print(True)
    else:
        print(False)
    if upper>0:
        print(True)
    else:
        print(False)

#Text Alignment
thickness = int(input()) #This must be an odd number
c = 'H'

    #Top Cone
    for i in range(thickness):
        print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

    #Top Pillars
    for i in range(thickness+1):
        print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

    #Middle Belt
    for i in range((thickness+1)//2):
        print((c*thickness*5).center(thickness*6))

    #Bottom Pillars
    for i in range(thickness+1):
        print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

    #Bottom Cone
    for i in range(thickness):
        print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

#Text Wrap
def wrap(string, max_width):
    output=[]
    for i,k in enumerate(string):
        if i % max_width == 0 and i > 0:
            output.append("\n")
        output.append(k)
    return ''.join(output)

#String Formatting
def print_formatted(number):
    s=len(bin(number)[2:])
    for i in range(1,number+1):
        print(str((i)).rjust(s), str(oct(i)[2:]).rjust(s), str(hex(i)[2:]).upper().rjust(s), str(bin(i)[2:]).rjust(s))

#Capitalize!
def solve(s):
    l = s.split(' ')
    res = []
    for w in l:
        res.append(w.capitalize())
    return ' '.join(res)

#The Minion Game
def minion_game(string):
    voc='AEIOU'
    stuart=0
    kevin=0
    lenst=len(string)
        
    for i, l in enumerate(string):
        if l in voc:
            kevin += lenst - i
        else:
            stuart += lenst - i
            
    if stuart > kevin:
        print('Stuart', stuart)
    elif kevin > stuart:
        print('Kevin', kevin)
    else:
        print('Draw')

#Merge the Tools!
def merge_the_tools(string, k):
    remaining_index = len(string)
    index_now = 0

    while remaining_index >= k:
        new_list = string[index_now:index_now+k]
        new_list = list(dict.fromkeys(new_list))
        print(''.join(new_list))
        index_now += k
        remaining_index -= k
    
    if remaining_index != 0:
        remaining_list = list(dict.fromkeys(string[index_now:]))
        print(''.join(remaining_list))

#SETS

#Introduction to Sets
def average(array):
    arr=set(array)
    return(sum(arr)/len(arr))

#No Idea!
n,m = input().split()
arr = input().split()
A = set(input().split())
B = set(input().split())

happy=0
for i in arr:
    if i in A:
        happy=happy+1
    if i in B:
        happy=happy-1
print(happy)

#Symmetric Difference
m=int(input())
mset=set(map(int,input().split()))
n=int(input())
nset=set(map(int,input().split()))

mn=list(mset.difference(nset))
nm=list(nset.difference(mset))
tot=mn+nm
tot.sort()
for i in tot:
    print(i)

#Set .add()
countryset=set()
for i in range(int(input())):
    countryset.add(input())
print(len(countryset))

#Set .discard(), .remove() & .pop()
n=int(input())
m=set(map(int,input().split()))
o=int(input())
for x in range(o):
    command=input().split()
    if command[0]=="pop":
        m.pop()
    elif command[0]=="remove":
        m.remove(int(command[1]))
    elif command[0]=="discard":
        m.discard(int(command[1]))
print(sum(m))

#Set .union() Operation
e=int(input())
en=set(map(int,input().split()))
f=int(input())
fr=set(map(int,input().split()))

print(len(en.union(fr)))

#Set .intersection() Operation
e=int(input())
en=set(map(int,input().split()))
f=int(input())
fr=set(map(int,input().split()))

print(len(en.intersection(fr)))

#Set .difference() Operation
e=int(input())
en=set(map(int,input().split()))
f=int(input())
fr=set(map(int,input().split()))

print(len(en.difference(fr)))

#Set .symmetric_difference() Operation
e=int(input())
en=set(map(int,input().split()))
f=int(input())
fr=set(map(int,input().split()))

print(len(en.symmetric_difference(fr)))

#Set Mutations
A=int(input())
Aset=set(map(int, input().split()))
for i in range(int(input())):
    op=list(input().split())
    nset=set(map(int, input().split()))
    if op[0]=='update':
        Aset.update(nset)
    if op[0]=='intersection_update':
        Aset.intersection_update(nset)
    if op[0]=='difference_update':
        Aset.difference_update(nset)
    if op[0]=='symmetric_difference_update':
        Aset.symmetric_difference_update(nset)

print(sum(Aset))

#The Captain's Room
from collections import Counter
n=int(input())
a=list(map(int,input().split()))
a=Counter(a)
for i in a:
    if a[i]!=n:
        print(i)

#Check Subset
for i in range(int(input())):
    a=int(input())
    A=set(map(int, input().split()))
    b=int(input())
    B=set(map(int, input().split()))
    if A.intersection(B)==A:
        print('True')
    else:
        print('False')

#Check Strict Superset
A=set(map(int, input().split()))
countset=0
numsubset=(int(input()))
for i in range(numsubset):
    s=set(map(int, input().split()))
    countnum=0
    for i in s:
        if (i in A)==True:
            countnum=countnum+1
    if countnum==len(s) and countnum<len(A):
        countset=countset+1
if countset==numsubset:
    print('True')
else:
    print('False')

#COLLECTIONS

#collections.Counter()
n=int(input())
l=list(map(int,input().split()))

customers=int(input())
earning=0
for i in range(customers):
    a,b=tuple(map(int,input().split()))
    if a in l:
        earning=earning+b
        l.remove(a)
print(earning)

#DefaultDict Tutorial
from collections import defaultdict
n,m= map(int, input().split())
d=defaultdict(list)
for i in range(n):
    d[input()].append(i+1)  
for i in range(m):
    x=input()
    if x in d:
        print(*d[x])
    else:
        print('-1')

#Collections.namedtuple()
from collections import namedtuple
num=int(input())
colname=input().split()
col=namedtuple('col', colname)
tot=0

for i in range(num):
    d=input().split()
    dati=col(ID=d[colname.index('ID')],MARKS=d[colname.index('MARKS')],NAME=d[colname.index('NAME')],CLASS=d[colname.index('CLASS')])
    tot=tot+int(dati.MARKS)

print("{:0.2f}".format(tot/num))

#Collections.OrderedDict()
n=int(input())
d={}
for i in range(n):
    a=input().split()
    item=" ".join(a[:-1])
    price=int(a[-1])
    d[item]=price+d.get(item, 0)
for i,j in d.items():
    print(i,j)

#Word Order
from collections import OrderedDict

d=OrderedDict()
for i in (input() for i in range(int(input()))):
    if i in d.keys():
        d[i] += 1
    else:
        d[i] = 1
        
print(len(d.keys()))
print(*d.values(), sep=' ')

#Collections.deque()
from collections import deque

d = deque()
for i in range(int(input())):
    a = input().split()
    if 'append' in a:
        d.append(a[1])
    elif 'appendleft' in a:
        d.appendleft(a[1])
    elif 'popleft' in a:
        d.popleft()
    elif 'pop' in a:
        d.pop()
print(*d)

#Company Logo
from collections import Counter

if __name__ == '__main__':
    s = input()
    c = Counter(sorted(s))
    for i,j in c.most_common(3):
        print(i,j)

#Piling Up!
from collections import deque
n = int(input())
for i in range(n):
    m = int(input())
    block = deque(map(int, input().split()))
    for j in range(m-1):
        if block[0] >= block[1] and block[0] >= block[-1]:
            block.popleft()
        elif block[-1] >= block[-2] and block[-1] >= block[0]:
            block.pop()
    if len(block) < 2:
        print('Yes')
    else:
        print('No')

#DATE AND TIME

#Calendar Module
import calendar as cal

m,d,y=map(int,input().split())
day=cal.weekday(year=y,month=m,day=d)
print(cal.day_name[day].upper())

#Time Delta
def time_delta(t1, t2):
    from datetime import datetime
    form='%a %d %b %Y %H:%M:%S %z'
    t1=datetime.strptime(t1, form)
    t2=datetime.strptime(t2, form)
    return str(int(abs(t1-t2).total_seconds()))

#EXCEPTIONS

#Exceptions
for i in range(int(input())):
    a,b=input().split()
    try:
        print(int(a)//int(b))
    except ZeroDivisionError as e:
        print('Error Code:',e)
        continue
    except ValueError as v:
        print('Error Code:',v)

#BUILT-INS

#Zipped!
n,m= map(int,input().split())
score=[]
for i in range(int(m)):
    scores=list(map(float, input().split()))
    score.append(scores)

for i in zip(*score):
    print(sum(i)/m)

#Athlete Sort
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    k = int(input())
    import operator
    newarray=sorted(arr, key=operator.itemgetter(k))
    for line in newarray:
        print(*line)

#ginortS
s=input()
low=[]
up=[]
odd=[]
even=[]
for i in range(len(s)):
    if s[i].islower():
        low.append(s[i])
    elif s[i].isupper():
        up.append(s[i])
    elif s[i].isdigit() and int(s[i]) % 2 > 0:
        odd.append(s[i])
    else:
        even.append(s[i])
low=sorted(low)
up=sorted(up)
odd=sorted(odd)
even=sorted(even)
print(''.join(low)+''.join(up)+''.join(odd)+''.join(even))

#FUNCTIONALS

#Map and Lambda Function
cube = lambda x: x**3

def fibonacci(n):
    lst = [0,1]
    if n>2:
        for i in range(2, n):
            lst.append(lst[i-1]+lst[i-2])
    else:
        lst = lst[0:n]
    return lst 

#REGEX AND PARSING

#Detect Floating Point Number
import re
T=int(input()) 
for i in range(T): 
    s = input()
    try:
        float(s)
        x = re.match(r"^[-+]?[0-9]*\.[0-9]+$", s)
        print(bool(x))
    except:
        print("False")

#Re.split()
regex_pattern = r"[,.]"

#Group(), Groups() & Groupdict()
import re
n = input()
pattern = re.compile(r"([\dA-Za-z])(?=\1)")
s = pattern.search(n)
if s:
    print(s.group(1))
else:
    print(-1)

#Re.findall() & Re.finditer()
import re
s = input()
p = "(?<=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])[aeiouAEIOU]{2,}(?=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])"
matches = re.findall(p, s)
if len(matches) != 0:
    print("\n".join(matches))
else:
    print("-1")

#Re.start() & Re.end()
import re
string= input()
pattern = re.compile(input())
match = pattern.search(string)
if not match: print("(-1, -1)")
while match:
    print(f"({match.start()}, {match.end()-1})")
    match = pattern.search(string,match.start() + 1)

#Regex Substitution
n=int(input())
import re
for i in range(n):
    a=re.sub(r'(\s\&\&)(?=(\s))',' and',input())
    print(re.sub(r'(\s\|\|)(?=(\s))',' or',a))

#Validating Roman Numerals
regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

#Validating phone numbers
import re
pattern=r"^[789][0-9]{9}$"
for i in range(int(input())):
    if re.search(pattern, input()):
        print('YES')
    else:
        print('NO')

#Validating and Parsing Email Addresses
import re
pattern = re.compile(r"<[a-z][a-zA-Z0-9\-\.\_]+\@[a-zA-Z]+\.[a-zA-Z]{1,3}>")
for i in range(int(input())):
    a = input().split()
    b = pattern.search(a[1])
    if b:
        print(a[0],b.string)

#Hex Color Code
import re
pattern = re.compile(r"[\s:](#[0-9A-Fa-f]{3,6})")
n = int(input())
for i in range(n):
    a=input()
    b=pattern.findall(a)
    for i in b:
        print(i)

#HTML Parser - Part 1
from html.parser import HTMLParser

class HParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start : "+tag)
        for attr in attrs:
            print("-> ", end = "")
            print(*attr, sep=" > ")
    def handle_endtag(self, tag):
        print("End   : "+tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty : "+tag)
        for attr in attrs:
            print("-> ", end = "")
            print(*attr, sep=" > ")

s = ""
for _ in range(int(input())):
    s += input()
parser = HParser()
parser.feed(s)

#HTML Parser - Part 2
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if '\n' in data:
            print(">>> Multi-line Comment")
            print(data)
        else:
            print(">>> Single-line Comment")
            print(data)
    def handle_data(self, data):
        if '\n' in data:
            data.strip()
        else:
            print(">>> Data")
            print(data)
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

#Detect HTML Tags, Attributes and Attribute Values
import re 
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        if attrs:
            for name,value in attrs:
                print(f"-> {name} > {value}")

parser = MyHTMLParser()
html = ''.join([input().strip() for _ in range(int(input()))])
parser.feed(html)

#Validating UID
T = int(input())
for i in range(T):
    uid = input()
    if len(uid) == 10 and len(set(uid)) == 10 and len([i for i in uid if i.isnumeric()]) >= 3 and len([i for i in uid if i.isupper()]) >= 2:
        print("Valid")
    else:
        print("Invalid")

#Validating Credit Card Numbers
import re

n=int(input())
for i in range(n):
    cc=input()
    if (re.fullmatch(r"^[456]\d{3}(-?\d{4}){3}$", cc) and \
         not re.search(r"([0-9])(-?\1){3}", cc)):
        print("Valid")
    else:
        print("Invalid")

#Validating Credit Card Numbers
import re

n=int(input())
for i in range(n):
    cc=input()
    if (re.fullmatch(r"^[456]\d{3}(-?\d{4}){3}$", cc) and \
         not re.search(r"([0-9])(-?\1){3}", cc)):
        print("Valid")
    else:
        print("Invalid")

#XML

#XML 1 - Find the Score
def get_attr_number(node):
    num=len(node.attrib)
    for i in node.findall('.//'):
        num += len(i.attrib)
    return num

#XML2 - Find the Maximum Depth
maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    if level > maxdepth:
        maxdepth = level
    for child in elem:
        depth(child, level)

#CLOSURES AND DECORATORS

#Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        lis=[]
        for item in l:
            lis.append('+91 '+item[-10:-5]+' '+ item[-5:])
        return f(lis)
    return fun

#Decorators 2 - Name Directory
def person_lister(f):
    def inner(people):
        for i in people:
            i[2]=int(i[2])
        people.sort(key=operator.itemgetter(2))
        return [f(person) for person in people]
  return inner

#NUMPY

#Mean, Var, and Std  #WRONG CODE
import numpy
numpy.set_printoptions(legacy='1.13')

n,m=map(int, input().split())
arr=numpy.array([list(map(int, input().split())) for i in range (n)])
print(numpy.mean((arr), axis=1))
print(numpy.var((arr), axis=0))
print(numpy.std(arr))

#Arrays
def arrays(arr):
    arr.reverse()
    return numpy.array(arr, float)

#Shape and Reshape
import numpy

arr=numpy.array(input().split(), int)
print(numpy.reshape(arr,(3,3)))

#Transpose and Flatten
import numpy

r,c=map(int, input().split())
l=[]
for i in range(r):
    l.append(list(map(int, input().split())))
arr=numpy.array(l)
print(numpy.transpose(arr))
print(arr.flatten())

#Concatenate
import numpy

n,m,p=map(int, input().split())
np=[]
mp=[]
for i in range(n):
    np.append(numpy.array(list(map(int, input().split()))))
for i in range(m):
    mp.append(numpy.array(list(map(int, input().split()))))
print(numpy.concatenate((np,mp), axis=0))

#Zeros and Ones
import numpy
dim=list(map(int,input().split()))
print(numpy.zeros((dim), dtype=numpy.int64))
print(numpy.ones((dim), dtype=numpy.int64))

#Eye and Identity
import numpy
numpy.set_printoptions(legacy='1.13')
n,m=map(int, input().split())
if n==m:
    print(numpy.identity(n))
else:
    print(numpy.eye(n,m))

#Array Mathematics
import numpy
n,m=map(int, input().split())
a=[numpy.array(input().split(), int) for i in range(n)]
b=[numpy.array(input().split(), int) for i in range(n)]
print(numpy.add(a,b))
print(numpy.subtract(a, b))
print(numpy.multiply(a, b))
print(numpy.floor_divide(a, b))
print(numpy.mod(a, b))
print(numpy.power(a, b))

#Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy='1.13')
arr=numpy.array(list(map(float,input().split())))
print(numpy.floor(arr))
print(numpy.ceil(arr))
print(numpy.rint(arr))

#Sum and Prod
import numpy

n,m=map(int, input().split())
arr=numpy.array([list(map(int, input().split())) for i in range (n)])
print(numpy.prod(numpy.sum((arr), axis=0)))

#Min and Max
import numpy

n,m=map(int, input().split())
arr=numpy.array([list(map(int, input().split())) for i in range (n)])
print(numpy.max(numpy.min(arr, axis=1)))

#Dot and Cross
import numpy

n=int(input())
arr1=numpy.array([list(map(int,input().split())) for i in range(n)])
arr2=numpy.array([list(map(int,input().split())) for i in range(n)])
print(numpy.dot(arr1,arr2))

#Inner and Outer
import numpy

a=numpy.array(list(map(int, input().split())))
b=numpy.array(list(map(int, input().split())))
print(numpy.inner(a,b))
print(numpy.outer(a,b))

#Polynomials
import numpy

p=list(map(float,input().split()))
x=float(input())
print(float(numpy.polyval(p,x)))

#Linear Algebra
import numpy
n=int(input())
arr=numpy.array([list(map(float, input().split())) for i in range(n)])
print(round(numpy.linalg.det(arr),2))

#EX 2

#Birthday Cake Candles
def birthdayCakeCandles(candles):
    tallest=max(candles)
    num=candles.count(tallest)
    return num

#Number Line Jumps
def kangaroo(x1, v1, x2, v2):
    if x1>x2 and v1>v2:
        return 'NO'
    elif x2>x1 and v2>v1:
        return 'NO'
    else:
        risp=0
        for i in range(1000000):
            x1=x1+v1
            x2=x2+v2
            if x1==x2:
                risp=1
                break
            else:
                continue
        if risp==1:
                return 'YES'
        else:
                return 'NO'

#Viral Advertising
def viralAdvertising(n):
    shared=5
    liked=2
    totlike=2
    for i in range(n-1):
        shared=liked*3
        liked=shared//2
        totlike=totlike+liked
    return totlike

#Compare the Triplets
def compareTriplets(a, b):
    alice=0
    bob=0
    for i in range(len(a)):
        if a[i]>b[i]:
            alice=alice+1
        elif b[i]>a[i]:
            bob=bob+1
    return[alice,bob]

#Insertion Sort - Part 1
def insertionSort1(n, arr):
    for i in range((n-1),0,-1):
        if arr[i] < arr[i-1]:
            sposta=arr[i]
            arr[i]=arr[i-1]
            print(*arr)
            arr[i-1]=sposta
    print(*arr)

#Insertion Sort - Part 2
def insertionSort2(n, arr):
    for i in range(1,n):
        for j in range(i):
            if arr[j] > arr[i]:
                arr[i], arr[j] = arr[j], arr[i]
        print(*arr)