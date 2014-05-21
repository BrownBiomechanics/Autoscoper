//! \file Vector.hpp
//! \author Andy Loomis (aloomis)

#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <cmath>
#include <iostream>

//! Forward declarations

template <typename T> struct Vec2;

template <typename T> struct Vec3;

template <typename T> struct Vec4;

//! Convenient typedefs for vectors

typedef Vec2<short> Vec2s;

typedef Vec2<int> Vec2i;

typedef Vec2<float> Vec2f;

typedef Vec2<double> Vec2d;

typedef Vec3<short> Vec3s;

typedef Vec3<int> Vec3i;

typedef Vec3<float> Vec3f;

typedef Vec3<double> Vec3d;

typedef Vec4<short> Vec4s;

typedef Vec4<int> Vec4i;

typedef Vec4<float> Vec4f;

typedef Vec4<double> Vec4d;

//! Class representing a 2-dimensional Euclidean vector

template <typename T>
struct Vec2
{
    //! Default constructor

    Vec2()
    {
        x = 0;
        y = 0;
    }

    //! Constructs a vector from the specified elements

    Vec2(const T& vx, const T& vy)
    {
        x = vx;
        y = vy;
    }

    //! Constructs a vector from the specified array

    template <typename U>
    explicit Vec2(const U* v)
    {
        x = v[0];
        y = v[1];
    }

    //! Copy constructor

    template <typename U>
    Vec2(const Vec2<U>& v)
    {
        x = v.x;
        y = v.y;
    }

    //! Constructs a vector from the specified vector

    template <typename U>
    Vec2(const Vec3<U>& v)
    {
        x = v.x;
        y = v.y;
    }

    //! Constructs a vector from the specified vector

    template <typename U>
    Vec2(const Vec4<U>& v)
    {
        x = v.x;
        y = v.y;
    }

    //! Returns a zero vector

    static const Vec2& zero()
    {
        static const Vec2 v;
        return v;
    }

    //! Returns a unit vector along the x-axis

    static const Vec2& unit_x()
    {
        static const Vec2 v(1,0);
        return v;
    }

    //! Returns a unit vector along the y-axis

    static const Vec2& unit_y()
    {
        static const Vec2 v(0,1);
        return v;
    }

    //! Swaps this vector and the specified vector

    void swap(Vec2& v)
    {
        std::swap(x,v.x);
        std::swap(y,v.y);
    }

    //! Assigns this vector to the specified vector

    template <typename U>
    Vec2& operator=(const Vec2<U>& v)
    {
        x = v.x;
        y = v.y;
        return *this;
    }

    //! Assigns this vector to the specified vector

    template <typename U>
    Vec2& operator=(const Vec3<U>& v)
    {
        x = v.x;
        y = v.y;
        return *this;
    }
    
    //! Assigns this vector to the specified vector

    template <typename U>
    Vec2& operator=(const Vec4<U>& v)
    {
        x = v.x;
        y = v.y;
        return *this;
    }

    //! Returns a pointer to the underlying array

    operator T*()
    {
        return data;
    }

    //! Returns a const pointer to the underlying array

    operator const T*() const
    {
        return data;
    }

    //! Returns a reference to the element at the specified index

    T& operator[](int i)
    {
        return data[i];
    }

    //! Returns a const reference to the element at the specified index

    const T& operator[](int i) const
    {
        return data[i];
    }

    //! Returns a reference to the element at the specified index

    T& operator()(int i)
    {
        return data[i];
    }

    //! Returns a const reference to the element at the specified index

    const T& operator()(int i) const
    {
        return data[i];
    }

    //! Returns the dot product of the specified vectors

    friend T dot(const Vec2& u, const Vec2& v)
    {
        return u.x*v.x+u.y*v.y;
    }

    //! Returns the squared length of the specified vector

    friend T lensq(const Vec2& v)
    {
        return dot(v,v);
    }

    //! Returns the length of the specified vector

    friend T len(const Vec2& v)
    {
        return sqrt(lensq(v));
    }

    //! Returns a unit vector in the direction of the specified vector.

    friend Vec2 unit(const Vec2& v)
    {
        return v/len(v);
    }

    //! Returns a copy of the specified vector

    friend Vec2 operator+(const Vec2& v)
    {
        return v;
    }

    //! Returns the negation of the specified vector

    friend Vec2 operator-(const Vec2& v)
    {
        return Vec2(-v.x,-v.y);
    }

    //! Returns the sum of the specified vectors

    friend Vec2 operator+(const Vec2& u, const Vec2& v)
    {
        return Vec2(u.x+v.x,u.y+v.y);
    }

    //! Returns the difference of the specified vectors

    friend Vec2 operator-(const Vec2& u, const Vec2& v)
    {
        return Vec2(u.x-v.x,u.y-v.y);
    }

    //! Returns the product by element of the specified vectors

    friend Vec2 operator*(const Vec2& u, const Vec2& v)
    {
        return Vec2(u.x*v.x,u.y*v.y);
    }

    //! Returns the quotient by element of the specified vectors

    friend Vec2 operator/(const Vec2& u, const Vec2& v)
    {
        return Vec2(u.x/v.x,u.y/v.y);
    }

    //! Returns the product of the specified scalar and vector

    friend Vec2 operator*(const T& s, const Vec2& v)
    {
        return Vec2(s*v.x,s*v.y);
    }

    //! Returns the product of the specified vector and scalar

    friend Vec2 operator*(const Vec2& v, const T& s)
    {
        return Vec2(v.x*s,v.y*s);
    }

    //! Returns the quotient of the specified vector and scalar

    friend Vec2 operator/(const Vec2& v, const T& s)
    {
        return Vec2(v.x/s,v.y/s);
    }

    //! Assigns a vector to the sum of the specified vectors

    friend Vec2& operator+=(Vec2& u, const Vec2& v)
    {
        u.x += v.x;
        u.y += v.y;
        return u;
    }

    //! Assigns a vector to the difference of the specified vectors

    friend Vec2& operator-=(Vec2& u, const Vec2& v)
    {
        u.x -= v.x;
        u.y -= v.y;
        return u;
    }

    //! Assigns a vector to the product by element of the specified vectors

    friend Vec2& operator*=(Vec2& u, const Vec2& v)
    {
        u.x *= v.x;
        u.y *= v.y;
        return u;
    }

    //! Assigns a vector to the quotient by element of the specified vectors

    friend Vec2& operator/=(Vec2& u, const Vec2& v)
    {
        u.x /= v.x;
        u.y /= v.y;
        return u;
    }

    //! Assigns a vector to the product of the specified vector and scalar

    friend Vec2& operator*=(Vec2& v, const T& s)
    {
        v.x *= s;
        v.y *= s;
        return v;
    }

    //! Assigns a vector to the quotient of the specified vector and scalar

    friend Vec2& operator/=(Vec2& v, const T& s)
    {
        v.x /= s;
        v.y /= s;
        return v;
    }

    //! Returns the minimum by element of the specified vectors

    friend Vec2 vmin(const Vec2& u, const Vec2& v)
    {
        return Vec2(((u.x < v.x)? u.x: v.x),((u.y < v.y)? u.y: v.y));
    }

    //! Returns the maximum by element of the specified vectors

    friend Vec2 vmax(const Vec2& u, const Vec2& v)
    {
        return Vec2(((u.x > v.x)? u.x: v.x),((u.y > v.y)? u.y: v.y));
    }

    //! Returns a linear interpolation of the specified vectors

    friend Vec2 lerp(const Vec2& u, const Vec2& v, const T& t)
    {
        return u+t*(v-u);
    }

    //! Returns true if the specified vectors are equal and false otherwise

    friend bool operator==(const Vec2& u, const Vec2& v)
    {
        return u.x == v.x && u.y == v.y;
    }

    //! Returns true if the specified vectors are unequal and false otherwise

    friend bool operator!=(const Vec2& u, const Vec2& v)
    {
        return u.x != v.x || u.y != v.y;
    }

    //! Writes the elements of the specified vector to the output stream

    friend std::ostream& operator<<(std::ostream& os, const Vec2& v)
    {
        return os << v.x << " " << v.y;
    }

    //! Reads the elements from the specified input stream to the vector

    friend std::istream& operator>>(std::ostream& is, Vec2& v)
    {
        return is >> v.x >> v.y;
    }

    //! Underlying data array
    
    union { struct { T x, y; }; T data[2]; };
};

//! Class representing a 3-dimensional Euclidean vector

template <typename T>
struct Vec3
{
    //! Default constructor

    Vec3()
    {
        x = 0;
        y = 0;
        z = 0;
    }

    //! Constructs a vector from the specified elements

    Vec3(const T& vx, const T& vy, const T& vz)
    {
        x = vx;
        y = vy;
        z = vz;
    }

    //! Constructs a vector from the specified array

    template <typename U>
    explicit Vec3(const U* v)
    {
        x = v[0];
        y = v[1];
        z = v[2];
    }

    //! Constructs a vector from the specified vector

    template <typename U>
    Vec3(const Vec2<U>& v, const T& vz = 0)
    {
        x = v.x;
        y = v.y;
        z = vz;
    }

    //! Copy constructor

    template <typename U>
    Vec3(const Vec3<U>& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    //! Constructs a vector from the specified vector

    template <typename U>
    Vec3(const Vec4<U>& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    //! Returns a zero vector

    static const Vec3& zero()
    {
        static const Vec3 v;
        return v;
    }

    //! Returns a unit vector along the x-axis

    static const Vec3& unit_x()
    {
        static const Vec3 v(1,0,0);
        return v;
    }

    //! Returns a unit vector along the y-axis

    static const Vec3& unit_y()
    {
        static const Vec3 v(0,1,0);
        return v;
    }

    //! Returns a unit vector along the z-axis

    static const Vec3& unit_z()
    {
        static const Vec3 v(0,0,1);
        return v;
    }

    //! Swaps this vector and the specified vector

    void swap(Vec3& v)
    {
        std::swap(x,v.x);
        std::swap(y,v.y);
        std::swap(z,v.z);
    }

    //! Assigns this vector to the specified vector

    template <typename U>
    Vec3& operator=(const Vec2<U>& v)
    {
        x = v.x;
        y = v.y;
        z = 0;
        return *this;
    }

    //! Assigns this vector to the specified vector

    template <typename U>
    Vec3& operator=(const Vec3<U>& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }

    //! Assigns this vector to the specified vector

    template <typename U>
    Vec3& operator=(const Vec4<U>& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }

    //! Returns a pointer to the underlying array

    operator T*()
    {
        return data;
    }

    //! Returns a const pointer to the underlying array

    operator const T*() const
    {
        return data;
    }

    //! Returns a reference to the element at the specified index

    T& operator[](int i)
    {
        return data[i];
    }

    //! Returns a const reference to the element at the specified index

    const T& operator[](int i) const
    {
        return data[i];
    }

    //! Returns a reference to the element at the specified index

    T& operator()(int i)
    {
        return data[i];
    }

    //! Returns a const reference to the element at the specified index

    const T& operator()(int i) const
    {
        return data[i];
    }

    //! Returns the dot product of the specified vectors

    friend T dot(const Vec3& u, const Vec3& v)
    {
        return u.x*v.x+u.y*v.y+u.z*v.z;
    }

    //! Returns the squared length of the specified vector

    friend T lensq(const Vec3& v)
    {
        return dot(v,v);
    }

    //! Returns the length of the specified vector

    friend T len(const Vec3& v)
    {
        return sqrt(lensq(v));
    }

    //! Returns a unit vector in the direction of the specified vector.

    friend Vec3 unit(const Vec3& v)
    {
        return v/len(v);
    }

    //! Returns a copy of the specified vector

    friend Vec3 operator+(const Vec3& v)
    {
        return v;
    }

    //! Returns the negation of the specified vector

    friend Vec3 operator-(const Vec3& v)
    {
        return Vec3(-v.x,-v.y,-v.z);
    }

    //! Returns the sum of the specified vectors

    friend Vec3 operator+(const Vec3& u, const Vec3& v)
    {
        return Vec3(u.x+v.x,u.y+v.y,u.z+v.z);
    }

    //! Returns the difference of the specified vectors

    friend Vec3 operator-(const Vec3& u, const Vec3& v)
    {
        return Vec3(u.x-v.x,u.y-v.y,u.z-v.z);
    }

    //! Returns the product by element of the specified vectors

    friend Vec3 operator*(const Vec3& u, const Vec3& v)
    {
        return Vec3(u.x*v.x,u.y*v.y,u.z*v.z);
    }

    //! Returns the quotient by element of the specified vectors

    friend Vec3 operator/(const Vec3& u, const Vec3& v)
    {
        return Vec3(u.x/v.x,u.y/v.y,u.z/v.z);
    }

    //! Returns the cross product of the specified vectors

    friend Vec3 cross(const Vec3& u, const Vec3& v)
    {
        return Vec3(u.y*v.z-u.z*v.y,u.z*v.x-u.x*v.z,u.x*v.y-u.y*v.x);
    }

    //! Returns the product of the specified scalar and vector

    friend Vec3 operator*(const T& s, const Vec3& v)
    {
        return Vec3(s*v.x,s*v.y,s*v.z);
    }

    //! Returns the product of the specified vector and scalar

    friend Vec3 operator*(const Vec3& v, const T& s)
    {
        return Vec3(v.x*s,v.y*s,v.z*s);
    }

    //! Returns the quotient of the specified vector and scalar

    friend Vec3 operator/(const Vec3& v, const T& s)
    {
        return Vec3(v.x/s,v.y/s,v.z/s);
    }

    //! Assigns a vector to the sum of the specified vectors

    friend Vec3& operator+=(Vec3& u, const Vec3& v)
    {
        u.x += v.x;
        u.y += v.y;
        u.z += v.z;
        return u;
    }

    //! Assigns a vector to the difference of the specified vectors

    friend Vec3& operator-=(Vec3& u, const Vec3& v)
    {
        u.x -= v.x;
        u.y -= v.y;
        u.z -= v.z;
        return u;
    }

    //! Assigns a vector to the product by element of the specified vectors

    friend Vec3& operator*=(Vec3& u, const Vec3& v)
    {
        u.x *= v.x;
        u.y *= v.y;
        u.z *= v.z;
        return u;
    }

    //! Assigns a vector to the quotient by element of the specified vectors

    friend Vec3& operator/=(Vec3& u, const Vec3& v)
    {
        u.x /= v.x;
        u.y /= v.y;
        u.z /= v.z;
        return u;
    }

    //! Assigns a vector to the product of the specified vector and scalar

    friend Vec3& operator*=(Vec3& v, const T& s)
    {
        v.x *= s;
        v.y *= s;
        v.z *= s;
        return v;
    }

    //! Assigns a vector to the quotient of the specified vector and scalar

    friend Vec3& operator/=(Vec3& v, const T& s)
    {
        v.x /= s;
        v.y /= s;
        v.z /= s;
        return v;
    }

    //! Returns the minimum by element of the specified vectors

    friend Vec3 vmin(const Vec3& u, const Vec3& v)
    {
        return Vec3(((u.x < v.x)? u.x: v.x),
                    ((u.y < v.y)? u.y: v.y),
                    ((u.z < v.z)? u.z: v.z));
    }

    //! Returns the maximum by element of the specified vectors

    friend Vec3 vmax(const Vec3& u, const Vec3& v)
    {
        return Vec3(((u.x > v.x)? u.x: v.x),
                    ((u.y > v.y)? u.y: v.y),
                    ((u.z > v.z)? u.z: v.z));
    }

    //! Returns a linear interpolation of the specified vectors

    friend Vec3 lerp(const Vec3& u, const Vec3& v, const T& t)
    {
        return u+t*(v-u);
    }

    //! Returns true if the specified vectors are equal and false otherwise

    friend bool operator==(const Vec3& u, const Vec3& v)
    {
        return u.x == v.x && u.y == v.y && u.z == v.z;
    }

    //! Returns true if the specified vectors are unequal and false otherwise

    friend bool operator!=(const Vec3& u, const Vec3& v)
    {
        return u.x != v.x || u.y != v.y || u.z != v.z;
    }

    //! Writes the elements of the specified vector to the output stream

    friend std::ostream& operator<<(std::ostream& os, const Vec3& v)
    {
        return os << v.x << " " << v.y << " " << v.z;
    }

    //! Reads the elements from the specified input stream to the vector

    friend std::istream& operator>>(std::istream& is, Vec3& v)
    {
        return is >> v.x >> v.y >> v.z;
    }

    //! Underlying data array
    
    union { struct { T x, y, z; }; T data[3]; };
};

//! Class representing a 4-dimensional Euclidean vector

template <typename T>
struct Vec4
{
    //! Default constructor

    Vec4()
    {
        x = 0;
        y = 0;
        z = 0;
        w = 0;
    }

    //! Constructs a vector from the specified elements

    Vec4(const T& vx, const T& vy, const T& vz, const T& vw)
    {
        x = vx;
        y = vy;
        z = vz;
        w = vw;
    }

    //! Constructs a vector from the specified array

    template <typename U>
    explicit Vec4(const U* v)
    {
        x = v[0];
        y = v[1];
        z = v[2];
        w = v[3];
    }

    //! Constructs a vector from the specified vector

    template <typename U>
    Vec4(const Vec2<U>& v, const T& vz = T(0), const T& vw = T(0))
    {
        x = v.x;
        y = v.y;
        z = vz;
        w = vw;
    }

    //! Constructs a vector from the specified vector

    template <typename U>
    Vec4(const Vec3<U>& v, const T& vw = T(0))
    {
        x = v.x;
        y = v.y;
        z = v.z;
        w = vw;
    }

    //! Copy constructor

    template <typename U>
    Vec4(const Vec4<U>& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        w = v.w;
    }

    //! Returns a zero vector

    static const Vec4& zero()
    {
        static const Vec4 v;
        return v;
    }

    //! Returns a unit vector along the x-axis

    static const Vec4& unit_x()
    {
        static const Vec4 v(1,0,0,0);
        return v;
    }

    //! Returns a unit vector along the y-axis

    static const Vec4& unit_y()
    {
        static const Vec4 v(0,1,0,0);
        return v;
    }

    //! Returns a unit vector along the z-axis

    static const Vec4& unit_z()
    {
        static const Vec4 v(0,0,1,0);
        return v;
    }

    //! Returns a unit vector along the w-axis

    static const Vec4& unit_w()
    {
        static const Vec4 v(0,0,0,1);
        return v;
    }

    //! Swaps this vector and the specified vector

    void swap(Vec4& v)
    {
        std::swap(x,v.x);
        std::swap(y,v.y);
        std::swap(z,v.z);
        std::swap(w,v.w);
    }

    //! Assigns this vector to the specified vector

    template <typename U>
    Vec4& operator=(const Vec2<U>& v)
    {
        x = v.x;
        y = v.y;
        z = 0;
        w = 0;
        return *this;
    }

    //! Assigns this vector to the specified vector

    template <typename U>
    Vec4& operator=(const Vec3<U>& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        w = 0;
        return *this;
    }

    //! Assigns this vector to the specified vector

    template <typename U>
    Vec4& operator=(const Vec4<U>& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        w = v.w;
        return *this;
    }

    //! Returns a pointer to the underlying array

    operator T*()
    {
        return data;
    }

    //! Returns a const pointer to the underlying array

    operator const T*() const
    {
        return data;
    }

    //! Returns a reference to the element at the specified index

    T& operator[](int i)
    {
        return data[i];
    }

    //! Returns a const reference to the element at the specified index

    const T& operator[](int i) const
    {
        return data[i];
    }

    //! Returns a reference to the element at the specified index

    T& operator()(int i)
    {
        return data[i];
    }

    //! Returns a const reference to the element at the specified index

    const T& operator()(int i) const
    {
        return data[i];
    }

    //! Returns the dot product of the specified vectors

    friend T dot(const Vec4& u, const Vec4& v)
    {
        return u.x*v.x+u.y*v.y+u.z*v.z+u.w*v.w;
    }

    //! Returns the squared length of the specified vector

    friend T lensq(const Vec4& v)
    {
        return dot(v,v);
    }

    //! Returns the length of the specified vector

    friend T len(const Vec4& v)
    {
        return sqrt(lensq(v));
    }

    //! Returns a unit vector in the direction of the specified vector.

    friend Vec4 unit(const Vec4& v)
    {
        return v/len(v);
    }

    //! Returns a copy of the specified vector

    friend Vec4 operator+(const Vec4& v)
    {
        return v;
    }

    //! Returns the negation of the specified vector

    friend Vec4 operator-(const Vec4& v)
    {
        return Vec4(-v.x,-v.y,-v.z,-v.w);
    }

    //! Returns the sum of the specified vectors

    friend Vec4 operator+(const Vec4& u, const Vec4& v)
    {
        return Vec4(u.x+v.x,u.y+v.y,u.z+v.z,u.w+v.w);
    }

    //! Returns the difference of the specified vectors

    friend Vec4 operator-(const Vec4& u, const Vec4& v)
    {
        return Vec4(u.x-v.x,u.y-v.y,u.z-v.z,u.w-v.w);
    }

    //! Returns the product by element of the specified vectors

    friend Vec4 operator*(const Vec4& u, const Vec4& v)
    {
        return Vec4(u.x*v.x,u.y*v.y,u.z*v.z,u.w*v.w);
    }

    //! Returns the quotient by element of the specified vectors

    friend Vec4 operator/(const Vec4& u, const Vec4& v)
    {
        return Vec4(u.x/v.x,u.y/v.y,u.z/v.z,u.w/v.w);
    }

    //! Returns the product of the specified scalar and vector

    friend Vec4 operator*(const T& s, const Vec4& v)
    {
        return Vec4(s*v.x,s*v.y,s*v.z,s*v.w);
    }

    //! Returns the product of the specified vector and scalar

    friend Vec4 operator*(const Vec4& v, const T& s)
    {
        return Vec4(v.x*s,v.y*s,v.z*s,v.w*s);
    }

    //! Returns the quotient of the specified vector and scalar

    friend Vec4 operator/(const Vec4& v, const T& s)
    {
        return Vec4(v.x/s,v.y/s,v.z/s,v.w/s);
    }

    //! Assigns a vector to the sum of the specified vectors

    friend Vec4& operator+=(Vec4& u, const Vec4& v)
    {
        u.x += v.x;
        u.y += v.y;
        u.z += v.z;
        u.w += v.w;
        return u;
    }

    //! Assigns a vector to the difference of the specified vectors

    friend Vec4& operator-=(Vec4& u, const Vec4& v)
    {
        u.x -= v.x;
        u.y -= v.y;
        u.z -= v.z;
        u.w -= v.w;
        return u;
    }

    //! Assigns a vector to the product by element of the specified vectors

    friend Vec4& operator*=(Vec4& u, const Vec4& v)
    {
        u.x *= v.x;
        u.y *= v.y;
        u.z *= v.z;
        u.w *= v.w;
        return u;
    }

    //! Assigns a vector to the quotient by element of the specified vectors

    friend Vec4& operator/=(Vec4& u, const Vec4& v)
    {
        u.x /= v.x;
        u.y /= v.y;
        u.z /= v.z;
        u.w /= v.w;
        return u;
    }

    //! Assigns a vector to the product of the specified vector and scalar

    friend Vec4& operator*=(Vec4& v, const T& s)
    {
        v.x *= s;
        v.y *= s;
        v.z *= s;
        v.w *= s;
        return v;
    }

    //! Assigns a vector to the quotient of the specified vector and scalar

    friend Vec4& operator/=(Vec4& v, const T& s)
    {
        v.x /= s;
        v.y /= s;
        v.z /= s;
        v.w /= s;
        return v;
    }

    //! Returns the minimum by element of the specified vectors

    friend Vec4 vmin(const Vec4& u, const Vec4& v)
    {
        return Vec4(((u.x < v.x)? u.x: v.x),
                    ((u.y < v.y)? u.y: v.y),
                    ((u.z < v.z)? u.z: v.z),
                    ((u.w < v.w)? u.w: v.w));
    }

    //! Returns the maximum by element of the specified vectors

    friend Vec4 vmax(const Vec4& u, const Vec4& v)
    {
        return Vec4(((u.x > v.x)? u.x: v.x),
                    ((u.y > v.y)? u.y: v.y),
                    ((u.z < v.z)? u.z: v.z),
                    ((u.w > v.w)? u.w: v.w));
    }

    //! Returns a linear interpolation of the specified vectors

    friend Vec4 lerp(const Vec4& u, const Vec4& v, const T& t)
    {
        return u+t*(v-u);
    }

    //! Returns true if the specified vectors are equal and false otherwise

    friend bool operator==(const Vec4& u, const Vec4& v)
    {
        return u.x == v.x && u.y == v.y && u.z == v.z && u.w == v.w;
    }

    //! Returns true if the specified vectors are unequal and false otherwise

    friend bool operator!=(const Vec4& u, const Vec4& v)
    {
        return u.x != v.x || u.y != v.y || u.z != v.z || u.w != v.w;
    }

    //! Writes the elements of the specified vector to the output stream

    friend std::ostream& operator<<(std::ostream& os, const Vec4& v)
    {
        return os << v.x << " " << v.y << " " << v.z << " " << v.w;
    }

    //! Reads the elements from the specified input stream to the vector

    friend std::istream& operator>>(std::istream& is, Vec4& v)
    {
        return is >> v.x >> v.y >> v.z >> v.w;
    }

    //! Underlying data array
    
    union { struct { T x, y, z, w; }; T data[4]; };
};

#endif // VECTOR_HPP
