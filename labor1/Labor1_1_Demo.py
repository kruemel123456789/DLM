# -*- coding: utf-8 -*-
"""
DLM LABOR 2.1: NumPy Demo
Wintersemester 2017
"""

import numpy as np
import matplotlib.pyplot as plt

"""
NumPy vereinfacht das Arbeiten mit Vektoren und Matrizen. 
Diese werden in NumPy als Arrays bezeichnet.
"""

Values = [1, 2, 3]
print(Values)
print( type(Values))

b = np.array([1, 2, 3])  
print( type(b))
print( b.dtype )

# Array aus der Liste erstellen: 
b = np.array(Values)
print( b )

""" 
np.array erwartet eine Liste:
a = np.array(1,2,3)    # Explodiert.
a = np.array([1,2,3])  # Funktioniert.
"""

b  = np.arange(1, 4, 0.5)
print( b )
print( b.shape )
print( b[0], b[1], b[2] )
b[0] = 5                 
print( b )                 

"""
Es gibt viele Wege ein Array zu erstellen.
"""

# Mit lists:
b = np.array([[1,2,3],[4,5,6]])  
print( b.shape )
print( b[0, 0], b[0, 1], b[1, 0] )
print( b )

# Ueber Horizontales/Vertikales Stapeln (Stacking)
b1 = [1,2,3]
b2 = [4,5,6]
b = np.vstack((b1,b2))
print( b.shape )

"""
stack-Funktionen erwarten ein Tupel:
b = np.vstack(b1,b2)    # Explodiert.
b = np.vstack((b1,b2))  # Funktioniert. 
"""

b1 = [1, 4]
b2 = [2, 5]
b3 = [3, 6]
b= np.hstack((b1,b2,b3))
print( b.shape )
print( b )#nope


b1 = np.array([1, 4])
b2 = np.array([2, 5])
b3 = np.array([3, 6])
b= np.hstack((b1,b2,b3))
print( b.shape )
print( b ) #nope

print( b1.shape )

b1 = np.array([[1], [4]])
b2 = np.array([[2], [5]])
b3 = np.array([[3], [6]])
b= np.hstack((b1,b2,b3))
print( b.shape )
print( b ) #geht


b1 = np.array([1, 4]).reshape(2,1)
b2 = np.array([2, 5]).reshape(2,-1) # eine dimension darf unbounded sein (-1)
b3 = np.array([3, 6]).T             # einfach Transponieren geht nicht
print( b3.shape )
b3 = np.array([3, 6]).reshape(1,2).T#
print( b3.shape )


b= np.hstack((b1,b2,b3))
print( b.shape )
print( b )#geht


b = np.arange(1,7).reshape(2,3)   
print( b.shape )

b = b.flatten()
print( b.shape )
print( b )

"""
Es gibt eine Reihe von speziellen Arrays
"""

c = np.zeros((4,2))
print( c )

c = np.ones_like(b)
print( c )

c = np.diag(np.array([1, 2, 3, 4, 5]))
print( c )
c = np.eye(4)
print( c )

c = np.kron(np.diag([1,2,3]), np.ones((2,2)))
print( c )

c = np.linspace(1, 2, 5)
print( c )


x = np.linspace(0, 1, 5)
y = np.arange(0, 1, 0.25)
xx, yy = np.meshgrid(x, y)

"""
Arrays und Datentypen
"""

b = np.array([1, 2, 3])  
print( type(b) )
print( b.dtype )

b = np.array([1., 2., 3.])  
print( type(b))
print( b.dtype )

b = np.array([1, 2, 3], dtype=np.float)  
print( type(b))
print( b.dtype )

b = np.diag(np.arange(1,6))
print( b.dtype )

b = np.ones_like(b)
print( b.dtype )

b = np.zeros(3)
print( b.dtype )

print( 3./2. )
print( 3/2 ) # neu mit python 3.x


"""
Einfache Matrix Operationen
"""

A = np.array([[1. ,2. ],[3. ,4. ]])
B = np.array([[5. ,6. ],[7. ,8. ]])



print( A+B )
print( np.add(A,B))

print( A-B )

print( A*B )
print( A/B )
print( np.sqrt(A) )

x = np.array([1, 2])
y = np.array([3, 4])

print( x.dot(y) )
print( np.dot(x, y))

print( x.shape )

y = y.reshape(1,2)
x = x.reshape(2,1)

print( x.dot(y) )
print( y.dot(x) )


x = np.array([1, 2])
y = np.array([3, 4])

print( A+x ) # Dieses Verhalten wird als Broadcasting bezeichnet. 
print( A.shape )
print( x.shape )

x = x.reshape(2,1)
print( A+x )

x = x.reshape(1,2)
print( A+x )

print( A*2 )

print( A.T )


"""
In arrays addressieren
"""
v = np.arange(10)

print( v[2] )

print( v[3:7] )

print( v[:] )

print( v[-1] )

print( v[3:10:3] )

print( v[3::3] )

print( v[::] )

print( v[:4] )

print( v[4:] )

v[-1] = 10
print( v )


b = np.arange(25).reshape(5,-1)  

print (b)
print( b.shape )

print( b[1,2] )

print( b[0] )


a = b[:,0]
print( a )
print( a.shape )
print( a[1] )


a = b[:1]
print( a )
print( a.shape )
print( a[1] )

print( b[3:, :3] )

print( b[:, ::3] )

print( b[:,[2, 3, 4, 1, 0]] )

boolIdx  = b > 10

print( boolIdx )

print( b[boolIdx])

print( b[b > 10] )


print( b.flat[10] )

print( b.flat[5:10] )
print( b.flat[5:10].shape )


b[0,:] = np.arange(4,-1,-1)
print( b )


"""
Views (Shallow/Deep Copy)
"""
A = np.zeros((5,5))
print( A )

B = A[1:4,1:4]
print( B )
type( B )

A[2, 2] = A[2, 2] +1
print( A )
print( B )

B[1]=1
print( A )
print( B )

B += 1
print( B )
print( A )

B = B+1
print( B )
print( A )

C = B
C += B
B[1,1] = 1
print( C )

"""
Random Data generieren

Plots von 2D Daten
"""

np.random.seed(1)
A = np.random.randn(2,2)
print( A )


x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y)
plt.show()
plt.plot(x,y, '--r')

