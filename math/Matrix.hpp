//! \file Matrix.hpp
//! \author Andy Loomis (aloomis)

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cmath>
#include <iostream>

#include "Quaternion.hpp"
#include "Vector.hpp"

//! Forward declarations

template <typename T> struct Mat2;

template <typename T> struct Mat3;

template <typename T> struct Mat4;

//! Convenient typedefs for matrices

typedef Mat2<float> Mat2f;

typedef Mat2<double> Mat2d;

typedef Mat3<float> Mat3f;

typedef Mat3<double> Mat3d;

typedef Mat4<float> Mat4f;

typedef Mat4<double> Mat4d;

//! Class representing a 2x2 transformation matrix

template <typename T>
struct Mat2
{
    //! Default constructor
	
    Mat2()
    {
		data[0][0] = T(1);
		data[0][1] = T(0);
		data[1][0] = T(0);
		data[1][1] = T(1);
	}

    //! Constructs a matrix from the specified elements
	
    Mat2(const T& m00, const T& m01, const T& m10, const T& m11)
    {
		data[0][0] = m00;
		data[0][1] = m01;
		data[1][0] = m10;
		data[1][1] = m11;
	}

    //! Constructs a matrix from the specified m
	
    explicit Mat2(const T* m)
    {
		data[0][0] = m[0];
		data[0][1] = m[1];
		data[1][0] = m[3];
		data[1][1] = m[4];
	}

    //! Constructs a matrix with the specified vectors as columns

    Mat2(const Vec2<T>& u, const Vec2<T>& v)
    {
        data[0][0] = u.x;
        data[0][1] = v.x;
        data[1][0] = u.y;
        data[1][1] = v.y;
    }

    //! Copy constructor
    
    Mat2(const Mat2& m)
    {
        data[0][0] = m(0,0);
        data[0][1] = m(0,1);
        data[1][0] = m(1,0);
        data[1][1] = m(1,1);
    }

    //! Constructs a matrix from the specified matrix
    
    explicit Mat2(const Mat3<T>& m)
    {
        data[0][0] = m(0,0);
        data[0][1] = m(0,1);
        data[1][0] = m(1,0);
        data[1][1] = m(1,1);
    }

    //! Constructs a matrix from the specified matrix
    
    explicit Mat2(const Mat4<T>& m)
    {
        data[0][0] = m(0,0);
        data[0][1] = m(0,1);
        data[1][0] = m(1,0);
        data[1][1] = m(1,1);
    }

    //! Returns a zero matrix

    static const Mat2& zero()
    {
        static const Mat2 m(T(0));
        return m;
    }

    //! Returns an indentity matrix

    static const Mat2& eye()
    {
        static const Mat2 m;
        return m;
    }

    //! Returns a scaleonal matrix
	
    static Mat2 scale(const T& sx, const T& sy)
    {
		return Mat2(sx,T(0),T(0),sy);
	}

    // Returns a rotation matrix
	
    static Mat2 rot(const T& rad)
    {
		T sin_rad = sin(rad);
		T cos_rad = cos(rad);
	    return Mat2(cos_rad,-sin_rad,sin_rad,cos_rad);
    }

    // Swaps this matrix with the specified matrix
    
    void swap(Mat2& m)
    {
        std::swap(data[0][0],m.data[0][0]);
        std::swap(data[0][1],m.data[0][1]);
        std::swap(data[1][0],m.data[1][0]);
        std::swap(data[1][1],m.data[1][1]);
    }

    // Assigns this matrix to the specified matrix
    
    Mat2& operator=(const Mat2& m)
    {
        data[0][0] = m.data[0][0];
        data[0][1] = m.data[0][1];
        data[1][0] = m.data[1][0];
        data[1][1] = m.data[1][1];
        return *this;
    }

    //! Returns a pointer to the underlying m
    
    operator T*()
    {
        return &data[0][0];
    }

    //! Returns a const pointer to the underlying m
    
    operator const T*() const
    {
        return &data[0][0];
    }

    //! Returns a pointer to the row at the specified index
    
    T* operator[](int i)
    {
        return &data[i][0];
    }

    //! Returns a const pointer to the row at the specified index
    
    const T* operator[](int i) const
    {
        return &data[i][0];
    }

    //! Returns a reference to the element at the specified indices
    
    T& operator()(int i, int j)
    {
        return data[i][j];
    }

    //! Returns a const reference to the element at the specified indices
    
    const T& operator()(int i, int j) const
    {
        return data[i][j];
    }

    //! Returns the specified row of the specified matrix

    friend Vec2<T> row(const Mat2& m, int i)
    {
        return Vec2<T>(m(i,0),m(i,1));
    }

    //! Returns the specified col of the specified matrix

    friend Vec2<T> col(const Mat2& m, int j)
    {
        return Vec2<T>(m(0,j),m(1,j));
    }
	
    //! Returns the determinant of the specified matrix
    
    friend T det(const Mat2& m)
    {
        return m(0,0)*m(1,1)-m(0,1)*m(1,0);
    }

    //! Returns the transpose of the specified matrix
    
    friend Mat2 trans(const Mat2& m)
    {
        return Mat2(m(0,0),m(1,0),m(0,1),m(1,1));
    }

    //! Returns the inverse of the specified matrix
    
    friend Mat2 inv(const Mat2& m)
    {
        return Mat2(m(1,1),-m(0,1),-m(1,0),m(0,0))/det(m);
    }

    //! Returns a copy of the specified matrix
    
    friend Mat2 operator+(const Mat2& m)
    {
        return m;
    }

    //! Returns the negation of the specified matrix
    
    friend Mat2 operator-(const Mat2& m)
    {
        return Mat2(-m(0,0),-m(0,1),-m(1,0),-m(1,1));
    }

    //! Returns the sum of the specified matricies
    
    friend Mat2 operator+(const Mat2& m, const Mat2& n)
    {
        return Mat2(m(0,0)+n(0,0),m(0,1)+n(0,1),
                    m(1,0)+n(1,0),m(1,1)+n(1,1));
    }

    //! Returns the difference of the specified matricies
    
    friend Mat2 operator-(const Mat2& m, const Mat2& n)
    {
        return Mat2(m(0,0)-n(0,0),m(0,1)-n(0,1),
                    m(1,0)-n(1,0),m(1,1)-n(1,1));
    }
    
    //! Returns the product of the specified matricies
    
    friend Mat2 operator*(const Mat2& m, const Mat2& n)
    {
        return Mat2(m(0,0)*n(0,0)+m(0,1)*n(1,0),m(0,0)*n(0,1)+m(0,1)*n(1,1),
                    m(1,0)*n(0,0)+m(1,1)*n(1,0),m(1,0)*n(0,1)+m(1,1)*n(1,1));
    }

    //! Returns the product of the specified matrix and vector
    
    friend Vec2<T> operator*(const Mat2& m, const Vec2<T>& v)
    {
        return Vec2<T>(m(0,0)*v.x+m(0,1)*v.y,m(1,0)*v.x+m(1,1)*v.y);
    }
	
    //! Returns the result of m*inv(n)
    
    friend Mat2 operator/(const Mat2& m, const Mat2& n)
    {
        return m*inv(n);
    }

    //! Returns the product of the specified scalar and matrix
    
    friend Mat2 operator*(const T& s, const Mat2& m)
    {
        return Mat2(s*m(0,0),s*m(0,1),s*m(1,0),s*m(1,1));
    }

    //! Returns the product of the specified matrix and scalar 
    
    friend Mat2 operator*(const Mat2& m, const T& s)
    {
        return Mat2(m(0,0)*s,m(0,1)*s,m(1,0)*s,m(1,1)*s);
    }

    //! Returns the quotient of the specified matrix and scalar
    
    friend Mat2 operator/(const Mat2& m, const T& s)
    {
        return Mat2(m(0,0)/s,m(0,1)/s,m(1,0)/s,m(1,1)/s);
    }

    //! Assigns a matrix to the sum of the specified matricies
    
    friend Mat2& operator+=(Mat2& m, const Mat2& n)
    {
        m(0,0) += n(0,0);
        m(0,1) += n(0,1);
        m(1,0) += n(1,0);
        m(1,1) += n(1,1);
        return m;
    }

    //! Assigns a matrix to the difference of the specified matricies
    
    friend Mat2& operator-=(Mat2& m, const Mat2& n)
    {
        m(0,0) -= n(0,0);
        m(0,1) -= n(0,1);
        m(1,0) -= n(1,0);
        m(1,1) -= n(1,1);
        return m;
    }

    //! Assigns a matrix to the product of the specified matricies
    
    friend Mat2& operator*=(Mat2& m, const Mat2& n)
    {
        return m = m*n;
    }

    //! Assigns a matrix to the result of m*inv(n)
    
    friend Mat2& operator/=(Mat2& m, const Mat2& n)
    {
        return m = m/n;
    }
   
    //! Assigns a matrix to the product of the specified matrix and scalar
    
    friend Mat2& operator*=(Mat2& m, const T& s)
    {
        m(0,0) *= s;
        m(0,1) *= s;
        m(1,0) *= s;
        m(1,1) *= s;
        return m;
    }

    //! Assigns a matrix to the quotient of the specified matrix and scalar
    
    friend Mat2& operator/=(Mat2& m, const T& s)
    {
        m(0,0) /= s;
        m(0,1) /= s;
        m(1,0) /= s;
        m(1,1) /= s;
        return m;
    }

    //! Returns true if the specified matrices are equal and false otherwise
    
    friend bool operator==(const Mat2& m, const Mat2& n)
    {
        return m(0,0) == n(0,0) && m(0,1) == n(0,1) && 
               m(1,0) == n(1,0) && m(1,1) == n(1,1);
    }

    //! Returns true if the specified matrices are not equal and false otherwise
    
    friend bool operator!=(const Mat2& m, const Mat2& n)
    {
        return m(0,0) != n(0,0) || m(0,1) != n(0,1) ||
               m(1,0) != n(1,0) || m(1,1) != n(1,1);
    }
   
    //! Writes the elements of the specified matrix to the output stream 
    
    friend std::ostream& operator<<(std::ostream& os, const Mat2& m)
    {
        return os << m(0,0) << " " << m(0,1) << " " << m(1,0) << " " << m(1,1);
    }

    //! Reads the elements from the specified input stream to the matrix
    
    friend std::istream& operator>>(std::istream& is, const Mat2& m)
    {
        return is >> m(0,0) >> m(0,1) >> m(1,0) >> m(1,1);
    }

    //! Underlying data m
	
    T data[2][2];
};

//! Class representing a 3x3 transformation matrix

template <typename T>
struct Mat3
{
    //! Default constructor
    
    Mat3()
    {
		data[0][0] = T(1);
		data[0][1] = T(0);
		data[0][2] = T(0);
		data[1][0] = T(0);
		data[1][1] = T(1);
		data[1][2] = T(0);
		data[2][0] = T(0);
		data[2][1] = T(0);
		data[2][2] = T(1);
	}

    //! Constructs a matrix from the specified elements
	
    Mat3(const T& m00, const T& m01, const T& m02,
         const T& m10, const T& m11, const T& m12,
         const T& m20, const T& m21, const T& m22)
    {
		data[0][0] = m00;
		data[0][1] = m01;
		data[0][2] = m02;
		data[1][0] = m10;
		data[1][1] = m11;
		data[1][2] = m12;
		data[2][0] = m20;
		data[2][1] = m21;
		data[2][2] = m22;
	}

    //! Constructs a matrix from the specified array
	
    explicit Mat3(const T* m)
    {
		data[0][0] = m[0];
		data[0][1] = m[1];
		data[0][2] = m[2];
		data[1][0] = m[3];
		data[1][1] = m[4];
		data[1][2] = m[5];
		data[2][0] = m[6];
		data[2][1] = m[7];
		data[2][2] = m[8];
	}

    //! Constructs a matrix with the specified vectors as columns

    Mat3(const Vec3<T>& u, const Vec3<T>& v, const Vec3<T>& n)
    {
		data[0][0] = u.x;
		data[0][1] = v.x;
		data[0][2] = n.x;
		data[1][0] = u.y;
		data[1][1] = v.y;
		data[1][2] = n.y;
		data[2][0] = u.z;
		data[2][1] = v.z;
		data[2][2] = n.z;
    }

    //! Constructs an affine transformation from the specified matrix
    
    explicit Mat3(const Mat2<T>& m, const Vec2<T>& v = Vec2<T>::zero())
    {
        data[0][0] = m(0,0);
        data[0][1] = m(0,1);
        data[0][2] = v.x;
        data[1][0] = m(1,0);
        data[1][1] = m(1,1);
        data[1][2] = v.y;
        data[2][0] = T(0);
        data[2][1] = T(0);
        data[2][2] = T(1);
    }
    
    //! Copy constructor
    
    Mat3(const Mat3& m)
    {
        data[0][0] = m(0,0);
        data[0][1] = m(0,1);
        data[0][2] = m(0,2);
        data[1][0] = m(1,0);
        data[1][1] = m(1,1);
        data[1][2] = m(1,2);
        data[2][0] = m(2,0);
        data[2][1] = m(2,1);
        data[2][2] = m(2,2);
    }

    //! Constructs a matrix from the specified matrix
    
    explicit Mat3(const Mat4<T>& m)
    {
        data[0][0] = m(0,0);
        data[0][1] = m(0,1);
        data[0][2] = m(0,2);
        data[1][0] = m(1,0);
        data[1][1] = m(1,1);
        data[1][2] = m(1,2);
        data[2][0] = m(2,0);
        data[2][1] = m(2,1);
        data[2][2] = m(2,2);
    }

    //! Returns a zero matrix

    static const Mat3& zero()
    {
        static const Mat3 m(T(0));
        return m;
    }

    //! Returns an indentity matrix

    static const Mat3& eye()
    {
        static const Mat3 m;
        return m;
    }

    //! Returns a scaleonal matrix
	
    static Mat3 scale(const T& sx, const T& sy, const T& sz)
    {
		return Mat3(sx,T(0),T(0),T(0),sy,T(0),T(0),T(0),sz);
	}

    //! Returns a matrix representing a rotation around the x-axis
	
    static Mat3 rot_x(const T& rad)
    {
		T sin_rad = sin(rad);
		T cos_rad = cos(rad);
		return Mat3(T(1),T(0),T(0),
                    T(0),cos_rad,-sin_rad,
                    T(0),sin_rad, cos_rad);
	}

    //! Returns a matrix representing a rotation around the y-axis
	
    static Mat3 rot_y(const T& rad)
    {
		T sin_rad = sin(rad);
		T cos_rad = cos(rad);
		return Mat3( cos_rad,T(0),sin_rad,
                     T(0),T(1),T(0),
                    -sin_rad,T(0),cos_rad);
	}

    //! Returns a matrix representing a rotation around the z-axis
	
    static Mat3 rot_z(const T& rad)
    {
		T sin_rad = sin(rad);
		T cos_rad = cos(rad);
		return Mat3(cos_rad,-sin_rad,T(0),
                    sin_rad, cos_rad,T(0),
                    T(0),T(0),T(1));
	}

    //! Returns a matrix representing a rotation around specified axis
	
    static Mat3 rot(const T& rad, const Vec3<T>& v)
    {
        T _lensq = lensq(v);
        if (_lensq < std::numeric_limits<T>::epsilon()) {
            return Mat3::eye();
        }

        Vec3<T> u = v/sqrt(_lensq);
        T sin_rad = sin(rad);
        T cos_rad = cos(rad);
        T one_minus_cos_rad = T(1)-cos_rad;
        return Mat3(u.x*u.x*one_minus_cos_rad+cos_rad,
                    u.x*u.y*one_minus_cos_rad-sin_rad*u.z,
                    u.x*u.z*one_minus_cos_rad+sin_rad*u.y,
                    u.y*u.x*one_minus_cos_rad+sin_rad*u.z,
                    u.y*u.y*one_minus_cos_rad+cos_rad,
                    u.y*u.z*one_minus_cos_rad-sin_rad*u.x,
                    u.z*u.x*one_minus_cos_rad-sin_rad*u.y,
                    u.z*u.y*one_minus_cos_rad+sin_rad*u.x,
                    u.z*u.z*one_minus_cos_rad+cos_rad);
    }

    //! Returns a matrix representing a rotation by the specified quaternion
    
    static Mat3 rot(const Quat<T>& q)
    {
        T _lensq = lensq(q);
        if (_lensq < std::numeric_limits<T>::epsilon()) {
            return Mat3::eye();
        }

        Quat<T> p = q/sqrt(_lensq);
        T xx = p.x*p.x;
        T xy = p.x*p.y;
        T xz = p.x*p.z;
        T xw = p.x*p.w;
        T yy = p.y*p.y;
        T yz = p.y*p.z;
        T yw = p.y*p.w;
        T zz = p.z*p.z;
        T zw = p.z*p.w;
        T ww = p.w*p.w;
        return Mat3(ww+xx-yy-zz,T(2)*(xy-zw),T(2)*(xz+yw),
                    T(2)*(xy+zw),ww-xx+yy-zz,T(2)*(yz-xw),
                    T(2)*(xz-yw),T(2)*(yz+xw),ww-xx-yy+zz);
    }

    //! Swaps this matrix with the specified matrix
    
    void swap(Mat3& m)
    {
        std::swap(data[0][0],m.data[0][0]);
        std::swap(data[0][1],m.data[0][1]);
        std::swap(data[0][2],m.data[0][2]);
        std::swap(data[1][0],m.data[1][0]);
        std::swap(data[1][1],m.data[1][1]);
        std::swap(data[1][2],m.data[1][2]);
        std::swap(data[2][0],m.data[2][0]);
        std::swap(data[2][1],m.data[2][1]);
        std::swap(data[2][2],m.data[2][2]);
    }

    //! Assigns this matrix to the specified matrix
    
    Mat3& operator=(const Mat3& m)
    {
        data[0][0] = m.data[0][0];
        data[0][1] = m.data[0][1];
        data[0][2] = m.data[0][2];
        data[1][0] = m.data[1][0];
        data[1][1] = m.data[1][1];
        data[1][2] = m.data[1][2];
        data[2][0] = m.data[2][0];
        data[2][1] = m.data[2][1];
        data[2][2] = m.data[2][2];
        return *this;
    }

    //! Returns a pointer to the underlying m
    
    operator T*()
    {
        return &data[0][0];
    }

    //! Returns a const pointer to the underlying m
    
    operator const T*() const
    {
        return &data[0][0];
    }

    //! Returns a pointer to the row at the specified index
    
    T* operator[](int i)
    {
        return &data[i][0];
    }

    //! Returns a const pointer to the row at the specified index
    
    const T* operator[](int i) const
    {
        return &data[i][0];
    }

    //! Returns a reference to the element at the specified indices
    
    T& operator()(int i, int j)
    {
        return data[i][j];
    }

    //! Returns a const reference to the element at the specified indices
    
    const T& operator()(int i, int j) const
    {
        return data[i][j];
    }

    //! Returns the specified row of the specified matrix

    friend Vec3<T> row(const Mat3& m, int i)
    {
        return Vec3<T>(m(i,0),m(i,1),m(i,2));
    }

    //! Returns the specified col of the specified matrix

    friend Vec3<T> col(const Mat3& m, int j)
    {
        return Vec3<T>(m(0,j),m(1,j),m(2,j));
    }
    
    //! Returns the determinant of the specified matrix
    
    friend T det(const Mat3& m)
    {
        return m(0,0)*m(1,1)*m(2,2)-m(0,0)*m(1,2)*m(2,1)-m(0,1)*m(1,0)*m(2,2)+
               m(0,1)*m(1,2)*m(2,0)+m(0,2)*m(1,0)*m(2,1)-m(0,2)*m(1,1)*m(2,0);
    }

    //! Returns the transpose of the specified matrix
    
    friend Mat3 trans(const Mat3& m)
    {
        return Mat3(m(0,0),m(1,0),m(2,0),
                    m(0,1),m(1,1),m(2,1),
                    m(0,2),m(1,2),m(2,2));
    }

    //! Returns the inverse of the specified matrix
    
    friend Mat3 inv(const Mat3& m)
    {
        // [ A ~ ~ ] [ ~ C ~ ] [ ~ ~ E ]
        // [ ~ B B ] [ D ~ D ] [ F F ~ ]
        // [ ~ B B ] [ D ~ D ] [ F F ~ ]

        T minor_b = m(1,1)*m(2,2)-m(1,2)*m(2,1);
        T minor_d = m(1,0)*m(2,2)-m(1,2)*m(2,0);
        T minor_F = m(1,0)*m(2,1)-m(1,1)*m(2,0);
        
        T det = m(0,0)*minor_b-m(0,1)*minor_d+m(0,2)*minor_F;
        
        return Mat3(minor_b/det,
                    (m(0,2)*m(2,1)-m(0,1)*m(2,2))/det,
                    (m(0,1)*m(1,2)-m(0,2)*m(1,1))/det,
                    -minor_d/det,
                    (m(0,0)*m(2,2)-m(0,2)*m(2,0))/det,
                    (m(0,2)*m(1,0)-m(0,0)*m(1,2))/det,
                    minor_F/det,
                    (m(0,1)*m(2,0)-m(0,0)*m(2,1))/det,
                    (m(0,0)*m(1,1)-m(0,1)*m(1,0))/det);
    }

    //! Returns a copy of the specified matrix
    
    friend Mat3 operator+(const Mat3& m)
    {
        return m;
    }

    //! Returns the negation of the specified matrix
    
    friend Mat3 operator-(const Mat3& m)
    {
        return Mat3(-m(0,0),-m(0,1),-m(0,2),
                    -m(1,0),-m(1,1),-m(1,2),
                    -m(2,0),-m(2,1),-m(2,2));
    }

    //! Returns the sum of the specified matricies
    
    friend Mat3 operator+(const Mat3& m, const Mat3& n)
    {
        return Mat3(m(0,0)+n(0,0),m(0,1)+n(0,1),m(0,2)+n(0,2),
                    m(1,0)+n(1,0),m(1,1)+n(1,1),m(1,2)+n(1,2),
                    m(2,0)+n(2,0),m(2,1)+n(2,1),m(2,2)+n(2,2));
    }

    //! Returns the difference of the specified matricies
    
    friend Mat3 operator-(const Mat3& m, const Mat3& n)
    {
        return Mat3(m(0,0)-n(0,0),m(0,1)-n(0,1),m(0,2)-n(0,2),
                    m(1,0)-n(1,0),m(1,1)-n(1,1),m(1,2)-n(1,2),
                    m(2,0)-n(2,0),m(2,1)-n(2,1),m(2,2)-n(2,2));
    }

    //! Returns the product of the specified matricies
    
    friend Mat3 operator*(const Mat3& m, const Mat3& n)
    {
        return Mat3(m(0,0)*n(0,0)+m(0,1)*n(1,0)+m(0,2)*n(2,0),
                    m(0,0)*n(0,1)+m(0,1)*n(1,1)+m(0,2)*n(2,1),
                    m(0,0)*n(0,2)+m(0,1)*n(1,2)+m(0,2)*n(2,2),
                    m(1,0)*n(0,0)+m(1,1)*n(1,0)+m(1,2)*n(2,0),
                    m(1,0)*n(0,1)+m(1,1)*n(1,1)+m(1,2)*n(2,1),
                    m(1,0)*n(0,2)+m(1,1)*n(1,2)+m(1,2)*n(2,2),
                    m(2,0)*n(0,0)+m(2,1)*n(1,0)+m(2,2)*n(2,0),
                    m(2,0)*n(0,1)+m(2,1)*n(1,1)+m(2,2)*n(2,1),
                    m(2,0)*n(0,2)+m(2,1)*n(1,2)+m(2,2)*n(2,2));
    }

    //! Returns the product of the specified matrix and vector
    
    friend Vec3<T> operator*(const Mat3& m, const Vec3<T>& v)
    {
        return Vec3<T>(m(0,0)*v.x+m(0,1)*v.y+m(0,2)*v.z,
                       m(1,0)*v.x+m(1,1)*v.y+m(1,2)*v.z,
                       m(2,0)*v.x+m(2,1)*v.y+m(2,2)*v.z);
    }
	
	//! Returns the point transformed by the specified matrix
	
	friend Vec2<T> mul_pt(const Mat3& m, const Vec2<T>& p)
	{
		T w = m(2,0)*p.x+m(2,1)*p.y+m(2,2);
		return Vec3<T>((m(0,0)*p.x+m(0,1)*p.y+m(0,2))/w,
					   (m(1,0)*p.x+m(1,1)*p.y+m(1,2))/w);
	}
	
	//! Returns the vector transformed by the specified matrix
	
	friend Vec3<T> mul_vec(const Mat3& m, const Vec2<T>& v)
	{
		return Vec3<T>(m(0,0)*v.x+m(0,1)*v.y,m(1,0)*v.x+m(1,1)*v.y);
	}
    
	//! Returns m*inv(n)
    
    friend Mat3 operator/(const Mat3& m, const Mat3& n)
    {
        return m*inv(n);
    }

    //! Returns the product of the specified scalar and matrix
    
    friend Mat3 operator*(const T& s, const Mat3& m)
    {
        return Mat3(s*m(0,0),s*m(0,1),s*m(0,2),
                    s*m(1,0),s*m(1,1),s*m(1,2),
                    s*m(2,0),s*m(2,1),s*m(2,2));
    }

    //! Returns the product of the specified matrix and scalar 
    
    friend Mat3 operator*(const Mat3& m, const T& s)
    {
        return Mat3(m(0,0)*s,m(0,1)*s,m(0,2)*s,
                    m(1,0)*s,m(1,1)*s,m(1,2)*s,
                    m(2,0)*s,m(2,1)*s,m(2,2)*s);
    }

    //! Returns the quotient of the specified matrix and scalar
    
    friend Mat3 operator/(const Mat3& m, const T& s)
    {
        return Mat3(m(0,0)/s,m(0,1)/s,m(0,2)/s,
                    m(1,0)/s,m(1,1)/s,m(1,2)/s,
                    m(2,0)/s,m(2,1)/s,m(2,2)/s);
    }

    //! Assigns a matrix to the sum of the specified matricies
    
    friend Mat3& operator+=(Mat3& m, const Mat3& n)
    {
        m(0,0) += n(0,0);
        m(0,1) += n(0,1);
        m(0,2) += n(0,2);
        m(1,0) += n(1,0);
        m(1,1) += n(1,1);
        m(1,2) += n(1,2);
        m(2,0) += n(2,0);
        m(2,1) += n(2,1);
        m(2,2) += n(2,2);
        return m;
    }

    //! Assigns a matrix to the difference of the specified matricies
    
    friend Mat3& operator-=(Mat3& m, const Mat3& n)
    {
        m(0,0) -= n(0,0);
        m(0,1) -= n(0,1);
        m(0,2) -= n(0,2);
        m(1,0) -= n(1,0);
        m(1,1) -= n(1,1);
        m(1,2) -= n(1,2);
        m(2,0) -= n(2,0);
        m(2,1) -= n(2,1);
        m(2,2) -= n(2,2);
        return m;
    }

    //! Assigns a matrix to the product of the specified matricies
    
    friend Mat3& operator*=(Mat3& m, const Mat3& n)
    {
        return m = m*n;
    }

    //! Assigns a matrix to the result of m*inv(n)
    
    friend Mat3& operator/=(Mat3& m, const Mat3& n)
    {
        return m = m/n;
    }
    
    //! Assigns a matrix to the product of the specified matrix and scalar
    
    friend Mat3& operator*=(Mat3& m, const T& s)
    {
        m(0,0) *= s;
        m(0,1) *= s;
        m(0,2) *= s;
        m(1,0) *= s;
        m(1,1) *= s;
        m(1,2) *= s;
        m(2,0) *= s;
        m(2,1) *= s;
        m(2,2) *= s;
        return m;
    }

    //! Assigns a matrix to the quotient of the specified matrix and scalar
    
    friend Mat3& operator/=(Mat3& m, const T& s)
    {
        m(0,0) /= s;
        m(0,1) /= s;
        m(0,2) /= s;
        m(1,0) /= s;
        m(1,1) /= s;
        m(1,2) /= s;
        m(2,0) /= s;
        m(2,1) /= s;
        m(2,2) /= s;
        return m;
    }

    //! Returns true if the specified matrices are equal and false otherwise
    
    friend bool operator==(const Mat3& m, const Mat3& n)
    {
        return m(0,0) == n(0,0) && m(0,1) == n(0,1) && m(0,2) == n(0,2) &&
               m(1,0) == n(1,0) && m(1,1) == n(1,1) && m(1,2) == n(1,2) &&
               m(2,0) == n(2,0) && m(2,1) == n(2,1) && m(2,2) == n(2,2);
    }

    //! Returns true if the specified matrices are not equal and false otherwise
    
    friend bool operator!=(const Mat3& m, const Mat3& n)
    {
        return m(0,0) != n(0,0) || m(0,1) != n(0,1) || m(0,2) != n(0,2) ||
               m(1,0) != n(1,0) || m(1,1) != n(1,1) || m(1,2) != n(1,2) ||
               m(2,0) != n(2,0) || m(2,1) != n(2,1) || m(2,2) != n(2,2);
    }
    
    //! Writes the elements of the specified matrix to the output stream 
    
    friend std::ostream& operator<<(std::ostream& os, const Mat3& m)
    {
        return os << m(0,0) << " " << m(0,1) << " " << m(0,2) << " "
                  << m(1,0) << " " << m(1,1) << " " << m(1,2) << " "
                  << m(2,0) << " " << m(2,1) << " " << m(2,2);
    }

    //! Reads the elements from the specified input stream to the matrix
    
    friend std::istream& operator>>(std::istream& is, const Mat3& m)
    {
        return is >> m(0,0) >> m(0,1) >> m(0,2)
                  >> m(1,0) >> m(1,1) >> m(1,2)
                  >> m(2,0) >> m(2,1) >> m(2,2);
    }

    //! Underlying data m
	
    T data[3][3];
};

//! Class representing a 4x4 transformation matrix

template <typename T>
struct Mat4
{
    //! Default constructor
    
    Mat4()
    {
		data[0][0] = T(1);
		data[0][1] = T(0);
		data[0][2] = T(0);
		data[0][3] = T(0);
		data[1][0] = T(0);
		data[1][1] = T(1);
		data[1][2] = T(0);
		data[1][3] = T(0);
		data[2][0] = T(0);
		data[2][1] = T(0);
		data[2][2] = T(1);
		data[2][3] = T(0);
		data[3][0] = T(0);
		data[3][1] = T(0);
		data[3][2] = T(0);
		data[3][3] = T(1);
	}

    //! Constructs a matrix from the specified elements
	
    Mat4(const T& m00, const T& m01, const T& m02, const T& m03,
         const T& m10, const T& m11, const T& m12, const T& m13,
         const T& m20, const T& m21, const T& m22, const T& m23,
         const T& m30, const T& m31, const T& m32, const T& m33)
    {
		data[0][0] = m00;
		data[0][1] = m01;
		data[0][2] = m02;
		data[0][3] = m03;
		data[1][0] = m10;
		data[1][1] = m11;
		data[1][2] = m12;
		data[1][3] = m13;
		data[2][0] = m20;
		data[2][1] = m21;
		data[2][2] = m22;
		data[2][3] = m23;
		data[3][0] = m30;
		data[3][1] = m31;
		data[3][2] = m32;
		data[3][3] = m33;
	}

    //! Constructs a matrix from the specified m

	explicit Mat4(const T* m)
    {
		data[0][0] = m[0];
		data[0][1] = m[1];
		data[0][2] = m[2];
		data[0][3] = m[3];
		data[1][0] = m[4];
		data[1][1] = m[5];
		data[1][2] = m[6];
		data[1][3] = m[7];
		data[2][0] = m[8];
		data[2][1] = m[9];
		data[2][2] = m[10];
		data[2][3] = m[11];
		data[3][0] = m[12];
		data[3][1] = m[13];
		data[3][2] = m[14];
		data[3][3] = m[15];
	}

    //! Constructs a matrix with the specified vectors as columns

    Mat4(const Vec4<T>& u, const Vec4<T>& v, const Vec4<T>& n, const Vec4<T>& t)
    {
		data[0][0] = u.x;
		data[0][1] = v.x;
		data[0][2] = n.x;
		data[0][3] = t.x;
		data[1][0] = u.y;
		data[1][1] = v.y;
		data[1][2] = n.y;
		data[1][3] = t.y;
		data[2][0] = u.z;
		data[2][1] = v.z;
		data[2][2] = n.z;
		data[2][3] = t.z;
		data[3][0] = u.w;
		data[3][1] = v.w;
		data[3][2] = n.w;
		data[3][3] = t.w;
    }

    //! Constructs a matrix from the specified matrix
    
    explicit Mat4(const Mat2<T>& m)
    {
        data[0][0] = m(0,0);
        data[0][1] = m(0,1);
        data[0][2] = T(0);
        data[0][3] = T(0);
        data[1][0] = m(1,0);
        data[1][1] = m(1,1);
        data[1][2] = T(0);
        data[1][3] = T(0);
        data[2][0] = T(0);
        data[2][1] = T(0);
        data[2][2] = T(1);
        data[2][3] = T(0);
        data[3][0] = T(0);
        data[3][1] = T(0);
        data[3][2] = T(0);
        data[3][3] = T(1);
    }

    //! Constructs an affine transformation from the specified matrix and vector
    
    explicit Mat4(const Mat3<T>& m, const Vec3<T>& v = Vec3<T>::zero())
    {
        data[0][0] = m(0,0);
        data[0][1] = m(0,1);
        data[0][2] = m(0,2);
        data[0][3] = v.x;
        data[1][0] = m(1,0);
        data[1][1] = m(1,1);
        data[1][2] = m(1,2);
        data[1][3] = v.y;
        data[2][0] = m(2,0);
        data[2][1] = m(2,1);
        data[2][2] = m(2,2);
        data[2][3] = v.z;
        data[3][0] = T(0);
        data[3][1] = T(0);
        data[3][2] = T(0);
        data[3][3] = T(1);
    }

    //! Copy constructor
    
    Mat4(const Mat4& m)
    {
        data[0][0] = m(0,0);
        data[0][1] = m(0,1);
        data[0][2] = m(0,2);
        data[0][3] = m(0,3);
        data[1][0] = m(1,0);
        data[1][1] = m(1,1);
        data[1][2] = m(1,2);
        data[1][3] = m(1,3);
        data[2][0] = m(2,0);
        data[2][1] = m(2,1);
        data[2][2] = m(2,2);
        data[2][3] = m(2,3);
        data[3][0] = m(3,0);
        data[3][1] = m(3,1);
        data[3][2] = m(3,2);
        data[3][3] = m(3,3);
    }

    //! Returns a zero matrix

    static const Mat4& zero()
    {
        static const Mat4 m(T(0));
        return m;
    }

    //! Returns an indentity matrix

    static const Mat4& eye()
    {
        static const Mat4 m;
        return m;
    }

	//! Returns a scaleonal matrix
	
	static Mat4 scale(const T& sx, const T& sy, const T& sz, const T& sw)
	{
		return Mat4(sx,T(0),T(0),T(0),
					T(0),sy,T(0),T(0),
					T(0),T(0),sz,T(0),
					T(0),T(0),T(0),sw);
	}
	
    //! Returns a perspective projection matrix
    
    static Mat4 persp(const T& fovy, const T& aspect,
					  const T& near, const T& far)
    {
        T top = near*tan(fovy/T(2));
        T right = aspect*top;
        return Mat4::frustum(-right,right,-top,top,near,far);
    }

    //! Returns a perspective projection matrix
    
    static Mat4 frustum(const T& left, const T& right,
                        const T& bottom, const T& top,
                        const T& near, const T& far)
    {
        T two_times_near = T(2)*near;
        T width = right-left;
        T height = top-bottom;
        T depth = far-near;
        return Mat4(two_times_near/width,T(0),(right+left)/width,T(0),
                    T(0),two_times_near/height,(top+bottom)/height,T(0),
                    T(0),T(0),-(far+near)/depth,-T(2)*far*near/depth,
                    T(0),T(0),-T(1),T(0));
    }

    //! Swaps this matrix with the specified matrix
    
    void swap(Mat4& m)
    {
        std::swap(data[0][0],m.data[0][0]);
        std::swap(data[0][1],m.data[0][1]);
        std::swap(data[0][2],m.data[0][2]);
        std::swap(data[0][3],m.data[0][3]);
        std::swap(data[1][0],m.data[1][0]);
        std::swap(data[1][1],m.data[1][1]);
        std::swap(data[1][2],m.data[1][2]);
        std::swap(data[1][3],m.data[1][3]);
        std::swap(data[2][0],m.data[2][0]);
        std::swap(data[2][1],m.data[2][1]);
        std::swap(data[2][2],m.data[2][2]);
        std::swap(data[2][3],m.data[2][3]);
        std::swap(data[3][0],m.data[3][0]);
        std::swap(data[3][1],m.data[3][1]);
        std::swap(data[3][2],m.data[3][2]);
        std::swap(data[3][3],m.data[3][3]);
    }

    //! Assigns this matrix to the specified matrix
    
    Mat4& operator=(const Mat4& m)
    {
        data[0][0] = m.data[0][0];
        data[1][0] = m.data[1][0];
        data[2][0] = m.data[2][0];
        data[3][0] = m.data[3][0];
        data[0][1] = m.data[0][1];
        data[1][1] = m.data[1][1];
        data[2][1] = m.data[2][1];
        data[3][1] = m.data[3][1];
        data[0][2] = m.data[0][2];
        data[1][2] = m.data[1][2];
        data[2][2] = m.data[2][2];
        data[3][2] = m.data[3][2];
        data[0][3] = m.data[0][3];
        data[1][3] = m.data[1][3];
        data[2][3] = m.data[2][3];
        data[3][3] = m.data[3][3];
        return *this;
    }

    //! Returns a pointer to the underlying m
    
    operator T*()
    {
        return &data[0][0];
    }

    //! Returns a const pointer to the underlying m
    
    operator const T*() const
    {
        return &data[0][0];
    }

    //! Returns a pointer to the row at the specified index
    
    T* operator[](int i)
    {
        return &data[i][0];
    }

    //! Returns a const pointer to the row at the specified index
    
    const T* operator[](int i) const
    {
        return &data[i][0];
    }

    //! Returns a reference to the element at the specified indices
    
    T& operator()(int i, int j)
    {
        return data[i][j];
    }

    //! Returns a const reference to the element at the specified indices
    
    const T& operator()(int i, int j) const
    {
        return data[i][j];
    }

    //! Returns the specified row of the specified matrix

    friend Vec4<T> row(const Mat4& m, int i)
    {
        return Vec4<T>(m(i,0),m(i,1),m(i,2),m(i,3));
    }

    //! Returns the specified col of the specified matrix

    friend Vec4<T> col(const Mat4& m, int j)
    {
        return Vec4<T>(m(0,j),m(1,j),m(2,j),m(3,j));
    }
    
    //! Returns the determinant of the specified matrix
    
    friend T det(const Mat4& m)
    {
        // [A A ~ ~]  [C ~ C ~]  [E ~ ~ E]
        // [A A ~ ~]  [C ~ C ~]  [E ~ ~ E]
        // [~ ~ B B]  [~ D ~ D]  [~ F F ~]
        // [~ ~ B B]  [~ D ~ D]  [~ F F ~]

        // [~ G G ~]  [~ I ~ I]  [~ ~ K K]
        // [~ G G ~]  [~ I ~ I]  [~ ~ K K]
        // [H ~ ~ H]  [J ~ J ~]  [L L ~ ~]
        // [H ~ ~ H]  [J ~ J ~]  [L L ~ ~]

        T minor_a = m(0,0)*m(1,1)-m(0,1)*m(1,0);
        T minor_b = m(2,2)*m(3,3)-m(2,3)*m(3,2);
        T minor_c = m(0,0)*m(1,2)-m(0,2)*m(1,0);
        T minor_d = m(2,1)*m(3,3)-m(2,3)*m(3,1);
        T minor_e = m(0,0)*m(1,3)-m(0,3)*m(1,0);
        T minor_F = m(2,1)*m(3,2)-m(2,2)*m(3,1);
        T minor_g = m(0,1)*m(1,2)-m(0,2)*m(1,1);
        T minor_h = m(2,0)*m(3,3)-m(2,3)*m(3,0);
        T minor_i = m(0,1)*m(1,3)-m(0,3)*m(1,1);
        T minor_j = m(2,0)*m(3,2)-m(2,2)*m(3,0);
        T minor_k = m(0,2)*m(1,3)-m(0,3)*m(1,2);
        T minor_l = m(2,0)*m(3,1)-m(2,1)*m(3,0);

        return minor_a*minor_b-minor_c*minor_d+minor_e*minor_F+
               minor_g*minor_h-minor_i*minor_j+minor_k*minor_l;
    }

    //! Returns the transpose of the specified matrix
    
    friend Mat4 trans(const Mat4& m)
    {
        return Mat4(m(0,0),m(1,0),m(2,0),m(3,0),
                    m(0,1),m(1,1),m(2,1),m(3,1),
                    m(0,2),m(1,2),m(2,2),m(3,2),
                    m(0,3),m(1,3),m(2,3),m(3,3));
    }

    //! Returns the inverse of the specified matrix
    
    friend Mat4 inv(const Mat4& m)
    {
        // [A A ~ ~]  [C ~ C ~]  [E ~ ~ E]
        // [A A ~ ~]  [C ~ C ~]  [E ~ ~ E]
        // [~ ~ B B]  [~ D ~ D]  [~ F F ~]
        // [~ ~ B B]  [~ D ~ D]  [~ F F ~]

        // [~ G G ~]  [~ I ~ I]  [~ ~ K K]
        // [~ G G ~]  [~ I ~ I]  [~ ~ K K]
        // [H ~ ~ H]  [J ~ J ~]  [L L ~ ~]
        // [H ~ ~ H]  [J ~ J ~]  [L L ~ ~]

        T minor_a = m(0,0)*m(1,1)-m(0,1)*m(1,0);
        T minor_b = m(2,2)*m(3,3)-m(2,3)*m(3,2);
        T minor_c = m(0,0)*m(1,2)-m(0,2)*m(1,0);
        T minor_d = m(2,1)*m(3,3)-m(2,3)*m(3,1);
        T minor_e = m(0,0)*m(1,3)-m(0,3)*m(1,0);
        T minor_F = m(2,1)*m(3,2)-m(2,2)*m(3,1);
        T minor_g = m(0,1)*m(1,2)-m(0,2)*m(1,1);
        T minor_h = m(2,0)*m(3,3)-m(2,3)*m(3,0);
        T minor_i = m(0,1)*m(1,3)-m(0,3)*m(1,1);
        T minor_j = m(2,0)*m(3,2)-m(2,2)*m(3,0);
        T minor_k = m(0,2)*m(1,3)-m(0,3)*m(1,2);
        T minor_l = m(2,0)*m(3,1)-m(2,1)*m(3,0);

        T det = minor_a*minor_b-minor_c*minor_d+minor_e*minor_F+
                minor_g*minor_h-minor_i*minor_j+minor_k*minor_l;

        return Mat4(( m(1,1)*minor_b-m(1,2)*minor_d+m(1,3)*minor_F)/det,
                    (-m(0,1)*minor_b+m(0,2)*minor_d-m(0,3)*minor_F)/det,
                    ( m(3,1)*minor_k-m(3,2)*minor_i+m(3,3)*minor_g)/det,
                    (-m(2,1)*minor_k+m(2,2)*minor_i-m(2,3)*minor_g)/det,
                    (-m(1,0)*minor_b+m(1,2)*minor_h-m(1,3)*minor_j)/det,
                    ( m(0,0)*minor_b-m(0,2)*minor_h+m(0,3)*minor_j)/det,
                    (-m(3,0)*minor_k+m(3,2)*minor_e-m(3,3)*minor_c)/det,
                    ( m(2,0)*minor_k-m(2,2)*minor_e+m(2,3)*minor_c)/det,
                    ( m(1,0)*minor_d-m(1,1)*minor_h+m(1,3)*minor_l)/det,
                    (-m(0,0)*minor_d+m(0,1)*minor_h-m(0,3)*minor_l)/det,
                    ( m(3,0)*minor_i-m(3,1)*minor_e+m(3,3)*minor_a)/det,
                    (-m(2,0)*minor_i+m(2,1)*minor_e-m(2,3)*minor_a)/det,
                    (-m(1,0)*minor_F+m(1,1)*minor_j-m(1,2)*minor_l)/det,
                    ( m(0,0)*minor_F-m(0,1)*minor_j+m(0,2)*minor_l)/det,
                    (-m(3,0)*minor_g+m(3,1)*minor_c-m(3,2)*minor_a)/det,
                    ( m(2,0)*minor_g-m(2,1)*minor_c+m(2,2)*minor_a)/det);
    }

    //! Returns a copy of the specified matrix
    
    friend Mat4 operator+(const Mat4& m)
    {
        return m;
    }

    //! Returns the negation of the specified matrix
    
    friend Mat4 operator-(const Mat4& m)
    {
        return Mat4(-m(0,0),-m(0,1),-m(0,2),-m(0,3),
                    -m(1,0),-m(1,1),-m(1,2),-m(1,3),
                    -m(2,0),-m(2,1),-m(2,2),-m(2,3),
                    -m(3,0),-m(3,1),-m(3,2),-m(3,3));
    }

    //! Returns the sum of the specified matricies
    
    friend Mat4 operator+(const Mat4& m, const Mat4& n)
    {
        return Mat4(m(0,0)+n(0,0),m(0,1)+n(0,1),m(0,2)+n(0,2),m(0,3)+n(0,3),
                    m(1,0)+n(1,0),m(1,1)+n(1,1),m(1,2)+n(1,2),m(1,3)+n(1,3),
                    m(2,0)+n(2,0),m(2,1)+n(2,1),m(2,2)+n(2,2),m(2,3)+n(2,3),
                    m(3,0)+n(3,0),m(3,1)+n(3,1),m(3,2)+n(3,2),m(3,3)+n(3,3));
    }

    //! Returns the difference of the specified matricies
    
    friend Mat4 operator-(const Mat4& m, const Mat4& n)
    {
        return Mat4(m(0,0)-n(0,0),m(0,1)-n(0,1),m(0,2)-n(0,2),m(0,3)-n(0,3),
                    m(1,0)-n(1,0),m(1,1)-n(1,1),m(1,2)-n(1,2),m(1,3)-n(1,3),
                    m(2,0)-n(2,0),m(2,1)-n(2,1),m(2,2)-n(2,2),m(2,3)-n(2,3),
                    m(3,0)-n(3,0),m(3,1)-n(3,1),m(3,2)-n(3,2),m(3,3)-n(3,3));
    }

    //! Returns the product of the specified matricies
    
    friend Mat4 operator*(const Mat4& m, const Mat4& n)
    {
        return Mat4(m(0,0)*n(0,0)+m(0,1)*n(1,0)+m(0,2)*n(2,0)+m(0,3)*n(3,0),
                    m(0,0)*n(0,1)+m(0,1)*n(1,1)+m(0,2)*n(2,1)+m(0,3)*n(3,1),
                    m(0,0)*n(0,2)+m(0,1)*n(1,2)+m(0,2)*n(2,2)+m(0,3)*n(3,2),
                    m(0,0)*n(0,3)+m(0,1)*n(1,3)+m(0,2)*n(2,3)+m(0,3)*n(3,3),
                    m(1,0)*n(0,0)+m(1,1)*n(1,0)+m(1,2)*n(2,0)+m(1,3)*n(3,0),
                    m(1,0)*n(0,1)+m(1,1)*n(1,1)+m(1,2)*n(2,1)+m(1,3)*n(3,1),
                    m(1,0)*n(0,2)+m(1,1)*n(1,2)+m(1,2)*n(2,2)+m(1,3)*n(3,2),
                    m(1,0)*n(0,3)+m(1,1)*n(1,3)+m(1,2)*n(2,3)+m(1,3)*n(3,3),
                    m(2,0)*n(0,0)+m(2,1)*n(1,0)+m(2,2)*n(2,0)+m(2,3)*n(3,0),
                    m(2,0)*n(0,1)+m(2,1)*n(1,1)+m(2,2)*n(2,1)+m(2,3)*n(3,1),
                    m(2,0)*n(0,2)+m(2,1)*n(1,2)+m(2,2)*n(2,2)+m(2,3)*n(3,2),
                    m(2,0)*n(0,3)+m(2,1)*n(1,3)+m(2,2)*n(2,3)+m(2,3)*n(3,3),
                    m(3,0)*n(0,0)+m(3,1)*n(1,0)+m(3,2)*n(2,0)+m(3,3)*n(3,0),
                    m(3,0)*n(0,1)+m(3,1)*n(1,1)+m(3,2)*n(2,1)+m(3,3)*n(3,1),
                    m(3,0)*n(0,2)+m(3,1)*n(1,2)+m(3,2)*n(2,2)+m(3,3)*n(3,2),
                    m(3,0)*n(0,3)+m(3,1)*n(1,3)+m(3,2)*n(2,3)+m(3,3)*n(3,3));
    }

    //! Returns the product of the specified matrix and vector
    
    friend Vec4<T> operator*(const Mat4& m, const Vec4<T>& v)
    {
        return Vec4<T>(m(0,0)*v.x+m(0,1)*v.y+m(0,2)*v.z+m(0,3)*v.w,
                       m(1,0)*v.x+m(1,1)*v.y+m(1,2)*v.z+m(1,3)*v.w,
                       m(2,0)*v.x+m(2,1)*v.y+m(2,2)*v.z+m(2,3)*v.w,
                       m(3,0)*v.x+m(3,1)*v.y+m(3,2)*v.z+m(3,3)*v.w);
    }
    
	//! Returns the point transformed by the specified matrix
	
	friend Vec3<T> mul_pt(const Mat4& m, const Vec3<T>& p)
	{
		T w = m(3,0)*p.x+m(3,1)*p.y+m(3,2)*p.z+m(3,3);
		return Vec3<T>((m(0,0)*p.x+m(0,1)*p.y+m(0,2)*p.z+m(0,3))/w,
					   (m(1,0)*p.x+m(1,1)*p.y+m(1,2)*p.z+m(1,3))/w,
					   (m(2,0)*p.x+m(2,1)*p.y+m(2,2)*p.z+m(2,3))/w);
	}
	
	//! Returns the vector transformed by the specified matrix
	
	friend Vec3<T> mul_vec(const Mat4& m, const Vec3<T>& v)
	{
		return Vec3<T>(m(0,0)*v.x+m(0,1)*v.y+m(0,2)*v.z,
					   m(1,0)*v.x+m(1,1)*v.y+m(1,2)*v.z,
					   m(2,0)*v.x+m(2,1)*v.y+m(2,2)*v.z);
	}
	
    //! Returns the result of m*inv(n)
    
    friend Mat4 operator/(const Mat4& m, const Mat4& n)
    {
        return m*inv(n);
    }

    //! Returns the product of the specified scalar and matrix
    
    friend Mat4 operator*(const T& s, const Mat4& m)
    {
        return Mat4(s*m(0,0),s*m(0,1),s*m(0,2),s*m(0,3),
                    s*m(1,0),s*m(1,1),s*m(1,2),s*m(1,3),
                    s*m(2,0),s*m(2,1),s*m(2,2),s*m(2,3),
                    s*m(3,0),s*m(3,1),s*m(3,2),s*m(3,3));
    }

    //! Returns the product of the specified matrix and scalar 
    
    friend Mat4 operator*(const Mat4& m, const T& s)
    {
        return Mat4(m(0,0)*s,m(0,1)*s,m(0,2)*s,m(0,3)*s,
                    m(1,0)*s,m(1,1)*s,m(1,2)*s,m(1,3)*s,
                    m(2,0)*s,m(2,1)*s,m(2,2)*s,m(2,3)*s,
                    m(3,0)*s,m(3,1)*s,m(3,2)*s,m(3,3)*s);
    }

    //! Returns the quotient of the specified matrix and scalar
    
    friend Mat4 operator/(const Mat4& m, const T& s)
    {
        return Mat4(m(0,0)/s,m(0,1)/s,m(0,2)/s,m(0,3)/s,
                    m(1,0)/s,m(1,1)/s,m(1,2)/s,m(1,3)/s,
                    m(2,0)/s,m(2,1)/s,m(2,2)/s,m(2,3)/s,
                    m(3,0)/s,m(3,1)/s,m(3,2)/s,m(3,3)/s);
    }

    //! Assigns a matrix to the sum of the specified matricies
    
    friend Mat4& operator+=(Mat4& m, const Mat4& n)
    {
        m(0,0) += n(0,0);
        m(0,1) += n(0,1);
        m(0,2) += n(0,2);
        m(0,3) += n(0,3);
        m(1,0) += n(1,0);
        m(1,1) += n(1,1);
        m(1,2) += n(1,2);
        m(1,3) += n(1,3);
        m(2,0) += n(2,0);
        m(2,1) += n(2,1);
        m(2,2) += n(2,2);
        m(2,3) += n(2,3);
        m(3,0) += n(3,0);
        m(3,1) += n(3,1);
        m(3,2) += n(3,2);
        m(3,3) += n(3,3);
        return m;
    }

    //! Assigns a matrix to the difference of the specified matricies
    
    friend Mat4& operator-=(Mat4& m, const Mat4& n)
    {
        m(0,0) -= n(0,0);
        m(0,1) -= n(0,1);
        m(0,2) -= n(0,2);
        m(0,3) -= n(0,3);
        m(1,0) -= n(1,0);
        m(1,1) -= n(1,1);
        m(1,2) -= n(1,2);
        m(1,3) -= n(1,3);
        m(2,0) -= n(2,0);
        m(2,1) -= n(2,1);
        m(2,2) -= n(2,2);
        m(2,3) -= n(2,3);
        m(3,0) -= n(3,0);
        m(3,1) -= n(3,1);
        m(3,2) -= n(3,2);
        m(3,3) -= n(3,3);
        return m;
    }

    //! Assigns a matrix to the product of the specified matricies
    
    friend Mat4& operator*=(Mat4& m, const Mat4& n)
    {
        return m = m*n;
    }

    //! Assigns a matrix to the result of m*inv(n)
    
    friend Mat4& operator/=(Mat4& m, const Mat4& n)
    {
        return m = m*inv(n);
    }
    
    //! Assigns a matrix to the product of the specified matrix and scalar
    
    friend Mat4& operator*=(Mat4& m, const T& s)
    {
        m(0,0) *= s;
        m(0,1) *= s;
        m(0,2) *= s;
        m(0,3) *= s;
        m(1,0) *= s;
        m(1,1) *= s;
        m(1,2) *= s;
        m(1,3) *= s;
        m(2,0) *= s;
        m(2,1) *= s;
        m(2,2) *= s;
        m(2,3) *= s;
        m(3,0) *= s;
        m(3,1) *= s;
        m(3,2) *= s;
        m(3,3) *= s;
        return m;
    }

    //! Assigns a matrix to the quotient of the specified matrix and scalar
    
    friend Mat4& operator/=(Mat4& m, const T& s)
    {
        m(0,0) /= s;
        m(0,1) /= s;
        m(0,2) /= s;
        m(0,3) /= s;
        m(1,0) /= s;
        m(1,1) /= s;
        m(1,2) /= s;
        m(1,3) /= s;
        m(2,0) /= s;
        m(2,1) /= s;
        m(2,2) /= s;
        m(2,3) /= s;
        m(3,0) /= s;
        m(3,1) /= s;
        m(3,2) /= s;
        m(3,3) /= s;
        return m;
    }

    //! Returns true if the specified matrices are equal and false otherwise
    
    friend bool operator==(const Mat4& m, const Mat4& n)
    {
        return m(0,0) == n(0,0) && m(0,1) == n(0,1) && m(0,2) == n(0,2) &&
               m(0,3) == n(0,3) && m(1,0) == n(1,0) && m(1,1) == n(1,1) &&
               m(1,2) == n(1,2) && m(1,3) == n(1,3) && m(2,0) == n(2,0) &&
               m(2,1) == n(2,1) && m(2,2) == n(2,2) && m(2,3) == n(2,3) &&
               m(3,0) == n(3,0) && m(3,1) == n(3,1) && m(3,2) == n(3,2) &&
               m(3,3) == n(3,3);
    }

    //! Returns true if the specified matrices are not equal and false otherwise
    
    friend bool operator!=(const Mat4& m, const Mat4& n)
    {
        return m(0,0) != n(0,0) || m(0,1) != n(0,1) || m(0,2) != n(0,2) ||
               m(0,3) != n(0,3) || m(1,0) != n(1,0) || m(1,1) != n(1,1) ||
               m(1,2) != n(1,2) || m(1,3) != n(1,3) || m(2,0) != n(2,0) ||
               m(2,1) != n(2,1) || m(2,2) != n(2,2) || m(2,3) != n(2,3) ||
               m(3,0) != n(3,0) || m(3,1) != n(3,1) || m(3,2) != n(3,2) ||
               m(3,3) != n(3,3);
    }
    
    //! Writes the elements of the specified matrix to the output stream 
    
    friend std::ostream& operator<<(std::ostream& os, const Mat4& m)
    {
        return os << m(0,0) << " " << m(0,1) << " " << m(0,2) << " "
                  << m(0,3) << " " << m(1,0) << " " << m(1,1) << " "
                  << m(1,2) << " " << m(1,3) << " " << m(2,0) << " "
                  << m(2,1) << " " << m(2,2) << " " << m(2,3) << " "
                  << m(3,0) << " " << m(3,1) << " " << m(3,2) << " "
                  << m(3,3);
    }

    //! Reads the elements from the specified input stream to the matrix
    
    friend std::istream& operator>>(std::istream is, Mat4& m)
    {
        return is >> m(0,0) >> m(0,1) >> m(0,2) >> m(0,3)
                  >> m(1,0) >> m(1,1) >> m(1,2) >> m(1,3)
                  >> m(2,0) >> m(2,1) >> m(2,2) >> m(2,3)
                  >> m(3,0) >> m(3,1) >> m(3,2) >> m(3,3);
    }

    //! Underlying data m
	
    T data[4][4];
};

#endif // MATRIX_HPP
