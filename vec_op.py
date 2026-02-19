import copy
from math import acos, sqrt
from typing import Self

import numpy as np
from rlbot.flat import Vector3

class OpVector3:
    def __init__(self, vector: Vector3) -> None:
        self._data = np.asarray([vector.x, vector.y, vector.z])
        
    @property
    def x(self) -> float:
        return self._data[0]
    
    @x.setter
    def x(self, value: float):
        self._data[0] = value
        
    @property
    def y(self) -> float:
        return self._data[1]
    
    @y.setter
    def y(self, value: float):
        self._data[1] = value
        
    @property
    def z(self) -> float:
        return self._data[2]
    
    @z.setter
    def z(self, value: float):
        self._data[2] = value
        
    def __getitem__(self, key):
        if isinstance(key, int):
            assert -2 <= key <= 2, f"Key must be -2 and 2, but {key} was given"
        if isinstance(key, slice):
            if key.start is not None:
                assert -2 <= key.start <= 2, f"Key slice start must be between -2 and 2, but {key.start} was given"
            
            if key.stop is not None:
                assert -2 <= key.stop <= 2, f"Key slice start must be between -2 and 2, but {key.start} was given"
            
        return self._data[key]
    
    def __neg__(self) -> "OpVector3":
        return OpVector3(
            Vector3(*-self._data)
        )
    
    def __add__(self, other: "Vector3 | OpVector3 | int | float") -> "OpVector3":
        _vec_copy = copy.copy(self)
        _vec_copy += other
        
        return _vec_copy
         
    def __sub__(self, other: "Vector3 | OpVector3 | int | float"):
        _vec_copy = copy.copy(self)
        _vec_copy -= other
        
        return _vec_copy
    
    def __mul__(self, other: "Vector3 | OpVector3 | int | float"):
        _vec_copy = copy.copy(self)
        _vec_copy *= other
        
        return _vec_copy
    
    def __truediv__(self, other: "Vector3 | OpVector3 | int | float"):
        _vec_copy = copy.copy(self)
        _vec_copy /= other
        
        return _vec_copy
    
    def __floordiv__(self, other: "Vector3 | OpVector3 | int | float"):
        _vec_copy = copy.copy(self)
        _vec_copy //= other
        
        return _vec_copy
    
    def __iadd__(self, other: "Vector3 | OpVector3 | int | float") -> Self:
        if isinstance(other, (Vector3, OpVector3)):
            self.x += other.x
            self.y += other.y
            self.z += other.z
            

        if isinstance(other, (int, float)):
            self._data += other
            
        return self
        
    def __isub__(self, other: "Vector3 | OpVector3 | int | float") -> Self:
        if isinstance(other, Vector3):
            value = Vector3(
                -other.x, -other.y, -other.z
            )
        else:
            value = -other
            
        self += value
        return self
    
    def __imul__(self, other: "Vector3 | OpVector3 | int | float") -> Self:        
        if isinstance(other, (Vector3, OpVector3)):
            self.x *= other.x
            self.y *= other.y
            self.z *= other.z

        if isinstance(other, (int, float)):
            self._data *= other
            
        return self
            
    def __itruediv__(self, other: "Vector3 | OpVector3 | int | float") -> Self:
        if isinstance(other, (Vector3, OpVector3)):
            value = other
            value.x = 1 / value.x
            value.y = 1 / value.y
            value.z = 1 / value.z
        else:
            value = 1 / other
            
        self *= value
        return self
    
    def __matmul__(self, other: "Vector3 | OpVector3"):
        _vec_copy = copy.copy(self)
        _vec_copy @= other
        
        return _vec_copy
    
    def __imatmul__(self, other: "Vector3 | OpVector3"):
        if isinstance(other, Vector3):
            value = OpVector3(other)
        else:
            value = other
            
        self._data = self._data @ value[:]
        return self
    
    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self._data))
    
    @property
    def normalized(self) -> "OpVector3":
        return self / self.magnitude
    
    def dot(self, vector: "OpVector3 | Vector3") -> float:
        if isinstance(vector, Vector3):
            value = OpVector3(vector)
        else:
            value = vector
        return (self @ value.transposed).magnitude
    
    def angle(self, vector: "OpVector3 | Vector3") -> float:
        if isinstance(vector, Vector3):
            value = OpVector3(vector)
        else:
            value = vector
        
        return acos(self.dot(value) / (self.magnitude * value.magnitude))
    
    def project_on(self, vector: "OpVector3 | Vector3") -> "OpVector3":
        if isinstance(vector, Vector3):
            value = OpVector3(vector)
        else:
            value = vector
            
        _normalized_vec = value.normalized
        _angle = self.angle(_normalized_vec)
        return _normalized_vec * (self.magnitude * _angle)
    
    def cross(self, vector: "OpVector3 | Vector3") -> "OpVector3":
        if isinstance(vector, Vector3):
            value = OpVector3(vector)
            
        else:
            value = vector
            
        return OpVector3(
            Vector3(
                self.y * value.z - self.z * value.y,
                self.z * value.x - self.x * value.z,
                self.x * value.y - self.y * value.x,
            )
        )
        
    @property
    def transposed(self) -> "OpVector3":
        return OpVector3.from_numpy(
            self._data.T
        )
    
    def to_vec3(self) -> Vector3:
        return Vector3(*self._data)
    
    def __str__(self) -> str:
        return "[" + ", ".join(map(str, self._data)) + "]"
    
    def __repr__(self) -> str:
        return f"x={self.x}, y={self.y}, z={self.z}"
    
    @staticmethod
    def from_numpy(numpy_array: np.ndarray) -> "OpVector3":
        assert numpy_array.shape == (3, ), f"Numpy array must be of shape (3, ), but received {numpy_array.shape}"
        
        return OpVector3(
            Vector3(*numpy_array)
        )