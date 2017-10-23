# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:22:39 2017

@author: mvetter
"""

# Zahlen
2 + 4

(100 - 3*5)/10

x = y = z = 0.0

x

# Komplexe Zahlen
2j * 3j

complex(2, 1) * complex(1, 1)

3 + 5j * 2

(1+2j) / (1+1j)

a = (10-5j)
print ( "Realteil: ", a.real, ", Imaginärteil: ", a.imag, ", Betrag: ", abs(a))

abs(a)

round(_,2)

#Strings
string1 = "Das ist ein String"
string2 = 'Das auch'
string3 = "C ist \"schneller\" als Python"
string3

print( string3 )

# Zeilenaufteilung eines Strings mit \
string4 =   "Das ist ein \
mehrzeiliger Text"
print( string4 )            
len( string4 )


# Strings werden mit + zusammengefügt
word = "Element" + ',' 
string = '<' + word * 3 + '>'	
print(string)

word[0]
word[1:]
word[-1]
word[:-2]

# Listen
list = ['DLM', 'Bachelor', 123, 1.0]
list
list[0]
list[1]

# Listeneintrag überschreiben
list[1] = 'Master'
2 * list[ :2] + [ 'Informationstechnik', 2.0 * 1008 ]

list = [1,5,8]
list[1: ] = [2,3]
list

# Liste einfügen
list[1:1] = [8, 8, 8]
list

#Elemente entfernen 
list[1:4] = []
list

# Kopie der Liste am Anfang einfügen
list[:0] = list
list

# Verschachtelte Listen
uList = [2.1, 2.2]
list = [1, uList, 3]
list
len(list)

print( list )
list[0]
list[1].append( 2.3 )
list
list[1]

# Zugriff auf Elemente der inneren Liste
list[1][0] = 2.0
list

# Initialisieren von Listen
print( range(5)  )
print(list)

list=[i for i in range(10)]
list

list=[i for i in range(5, 15, 2)]
list

list=[ i * i + 1 for i in range(10)]	
list

list=[i*i for i in [3, 5, 10]]
list

list2=[x for x in list if x >= 20]
list2

[(x, x*x) for x in list2 if x >= 5]

from math import pi
[str(round(pi, i)) for i in range(1,10)]

list
x = 10

list.append(x)	# neues Element an das Ende
list
list.extend(list2)	# hängt die list2 an das Ende list[len(list):] = list2
list
list.insert(0,x)	# list.insert(0, x) fügt ein Element an den Anfang der Liste
list
list.remove(x)	# entfernt alle x aus der liste
list
list.pop()		# entfernt das letzte Element
list
list.index(25)	# liefert den Index von x 
list.count(100)	# zählt das Vorkommen von x
list.sort()		# sortiert die Liste
list
list.reverse()	# kehrt die Reihenfolge der Liste um
list
del list[2:5]	# entfernt die Elemente von der Position 2 bis 4
list

# Initialisierung von zweidimensionalen Listen
list=[[1,0],[0,1]]
list

list=[[i+i for i in range(3)] for j in range(4)] 
list			

list=[[[i+j+k for i in range(3)] for j in range(4)] for k in range(5)]
list			

# Tupel
t = 123, 456, 'Text'
t

# verschachtelte tupel
u = t, (10, 20, 30)
u
u[1][0]

# Mengen
korb1 = {'Moehren', 'Kartoffeln', 'Blumenkohl', 'Chicoree'}
korb1 
'Blumenkohl' in korb1

korb2 = {'Bananen', 'Orangen', 'Chicoree'}
korb2
korb1-korb2
korb1|korb2
korb1 & korb2
korb1 ^ korb2


m1 = {x for x in 'abracadabra' }
m1
m2 = {x for x in 'abracadabra' if x not in 'abc'}	
m2

# Dictionaries
matNr =  { 'Bernd': 1455314, 'Petra': 1453551}
matNr
matNr[ 'Bernd' ]

del matNr['Bernd']
matNr

matNr[ 'Hans'] =  1455315
matNr

'Petra' in matNr

{x: x**2 for x in (2, 4, 6)}

# Schleifen
a, b = 0, 1
while b < 100:
    print(b)
    a, b = b, a+b



list = ['PKW', 'LKW', 'Motorrad']
for x in list:
    print( x, 'hat', len(x), 'Zeichen' )		


# Die Funktion Rage
for i in range(5):
    print(i)
    
for i in range(5, 10):
	print(i)  
    
for i in range(0, 10, 3):
	print(i)    
    
for i in range(-1, -10, -2):
	print(i)    

a = ['Hallo', 'Welt', 'in', '2017' ]	
for i in range(0, len(a), 2):	
    print(i, a[i])	



# If Else
x = int( input('Bitte geben Sie einen Integer ein: ') )
x

if x < 0:
	x = 1
	print('kleiner Null wird zur Eins')
elif x == 0:
	print('Null')
elif x == 1:
	print('Eins')
else:
	print('Mehr')
    
    
# Funktionen definieren    

def fib(n):    	
	"""Gibt die Fibonacci-Folge bis n aus"""
	a, b = 0, 1
	while a < n:
		print(a, end=' ')
		a, b = b, a+b

f = fib
f(99)


# Funktionsparameter werden als call by value übergeben
def fib(n=100):    	
	"""Gibt die Fibonacci-Folge bis n aus"""
	result = []
	a, b = 0, 1
	while a < n:
		result.append(a)
		a, b = b, a+b
	return result

fib10 = fib(10)
fib10
fib()

# Klassen
class MyClass:
    """A simple example class"""
    i = 123
    def f(self):
        return 'Hallo Welt'
    
my = MyClass()
my.f()

# 
class MyComplex:
    real = 0
    imag = 0        
    def __init__(self, r, i):
        self.real = r
        self.imag = i       
    def print(self):
        print( 'rel: ', self.real, 'imag:', self.imag)		

x = MyComplex( 10,-5.5)
x.print()

#  Leere Klassen
class Schueler:
    pass

class Student:
    pass

class Lehrer:
    pass

class MyClass(Schueler, Student, Lehrer):
    _VariablenOderFunktionenDieMitUnterstrichBeginnenGeltenAlsPrivat = 0



class Student:
    pass

hans = Student

hans.name = " Hans Mayer"
hans.kurs = 'DLM'
hans.praktikum1 = True

hans.name

# Lesen und schreiben von Dateien
import sys
f = open('C:/gitWorkspace/vorlesung\dlm/praktikum/praktikum1/test.txt', 'r')
f.read()		
f.seek(0)
f.readline()            # ‘zeile1\n‘
list = f.readlines()	   # [‘zeile1\n‘, ‘zeile2\n‘, ‘zeile3\n‘]
f.close()
list

f = open('C:/gitWorkspace/vorlesung\dlm/praktikum/praktikum1/test2.txt', 'w')
string = "Alle Beispiele ausprobieren"
f.write( string )
f.close()




