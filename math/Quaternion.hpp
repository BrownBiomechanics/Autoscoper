//! \file Quat.hpp
//! \author Andy Loomis (aloomis)

#ifndef QUATERNION_HPP
#define QUATERNION_HPP

#include <cmath>
#include <iostream>
#include <limits>

#include "Matrix.hpp"
#include "Vector.hpp"

//! Forward declarations

template <typename T> struct Mat3;

template <typename T> struct Quat;

//! Convenient typedefs for quaternions

typedef Quat<float> Quatf;

typedef Quat<double> Quatd;

//! Class representing a quaternion

template <typename T>
struct Quat
{
    //! Defualt constructor

    Quat()
    {
        w = T(1);
        x = T(0);
        y = T(0);
        z = T(0);
    }

    //! Constructs a quaternion from the specified elements

    Quat(const T& qw, const T& qx, const T& qy, const T& qz)
    {
        w = qw;
        x = qx;
        y = qy;
        z = qz;
    }

    //! Constructs a quaternion from the specified array

    explicit Quat(const T* q)
    {
        w = q->w;
        x = q->x;
        y = q->y;
        z = q->z;
    }

    //! Returns a zero quaternion

    static const Quat& zero()
    {
        static const Quat q(T(0));
        return q;
    }

    //! Returns a quaternion representing an eye rotation

    static const Quat& eye()
    {
        static const Quat q;
        return q;
    }

    //! Returns a quaternion representing a rotation by the specified angle
    //! about the specified vector

    static Quat rot(const T& rad, const Vec3<T>& v)
    {
        T _lensq = lensq(v);
        if (_lensq < std::numeric_limits<T>::epsilon()) {
            return Quat::eye();
        }

        T sin_rad_over_len = sin(rad/T(2))/sqrt(_lensq);
        return Quat(cos(rad/T(2)),
                    sin_rad_over_len*v.x,
                    sin_rad_over_len*v.y,
                    sin_rad_over_len*v.z);
    }

    //! Returns a quaternion representing a rotation by the specified matrix

    static Quat rot(const Mat3<T>& m)
    {
        int i = m(1,1) > m(0,0)? 1: 0; if (m(2,2) > m(i,i)) { i = 2; }
        int j = (i+1)%3;
        int k = (j+1)%3;

        T r = T(2)*sqrt(T(1)+m(i,i)-m(j,j)-m(k,k));
        if (r < std::numeric_limits<T>::epsilon()) {
            return Quat::eye();
        }

        if (i == 0) {
            return Quat((m(2,1)-m(1,2))/r,
                        r/T(4),
                        (m(0,1)+m(1,0))/r,
                        (m(2,0)+m(0,2))/r);
        }
        else if (i == 1) {
            return Quat((m(0,2)-m(2,0))/r,
                        (m(0,1)+m(1,0))/r,
                        r/T(4),
                        (m(1,2)+m(2,1))/r);
        }
        else {
            return Quat((m(1,0)-m(0,1))/r,
                        (m(2,0)+m(0,2))/r,
                        (m(1,2)+m(2,1))/r,
                        r/T(4));
        }
    }

    //! Swaps this quaternion with the specified quaternion

    void swap(const Quat& q)
    {
        std::swap(w,q.w);
        std::swap(x,q.x);
        std::swap(y,q.y);
        std::swap(z,q.z);
    }

    //! Assigns this quaternion to the specified quaternion

    Quat& operator=(const Quat& q)
    {
        w = q.w;
        x = q.x;
        y = q.y;
        z = q.z;
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

    //! Returns a vector representing the axis of rotation of the specified
    //! quaternion

    friend Vec3<T> axis(const Quat& q)
    {
        T len = sqrt(q.x*q.x+q.y*q.y+q.z*q.z);
        return Vec3<T>(q.x/len,q.y/len,q.z/len);
    }

    //! Returns the angle of rotation of the specified quaternion

    friend T angle(const Quat& q)
    {
        return T(2)*acos(q.w);
    }

    //! Returns the dot product of the specified quaternions

    friend T dot(const Quat& p, const Quat& q)
    {
        return p.w*q.w+p.x*q.x+p.y*q.y+p.z*q.z;
    }

    //! Returns the squared length of the specified quaternion

    friend T lensq(const Quat& q)
    {
        return dot(q,q);
    }

    //! Returns the length of the specified quaternion

    friend T len(const Quat& q)
    {
        return sqrt(lensq(q));
    }

    //! Returns a unit quaternion

    friend Quat unit(const Quat& q)
    {
        return q/len(q);
    }

    //! Returns the conjugate of the specified quaternion

    friend Quat conj(const Quat& q)
    {
        return Quat(q.w,-q.x,-q.y,-q.z);
    }

    //! Returns the inverse of the specified quaternion

    friend Quat inv(const Quat& q)
    {
        return conj(q)/lensq(q);
    }

    //! Returns a copy of the specified quaternion

    friend Quat operator+(const Quat& q)
    {
        return q;
    }

    //! Returns the negation of the specified quaternion

    friend Quat operator-(const Quat& q)
    {
        return Quat(-q.w,-q.x,-q.y,-q.z);
    }

    //! Returns the sum of the specified quaternions

    friend Quat operator+(const Quat& p, const Quat& q)
    {
        return Quat(p.w+q.w,p.x+q.x,p.y+q.y,p.z+q.z);
    }

    //! Returns the difference of the specified quaternions

    friend Quat operator-(const Quat& p, const Quat& q)
    {
        return Quat(p.w-q.w,p.x-q.x,p.y-q.y,p.z-q.z);
    }

    //! Returns the product of the specified quaternions

    friend Quat operator*(const Quat& p, const Quat& q)
    {
        return Quat(p.w*q.y-p.x*q.z+p.y*q.w+p.z*q.x,
                    p.w*q.z+p.x*q.y-p.y*q.x+p.z*q.w,
                    p.w*q.w-p.x*q.x-p.y*q.y-p.z*q.z,
                    p.w*q.x+p.x*q.w+p.y*q.z-p.z*q.y);
    }

    //! Returns the result of p*inv(q)

    friend Quat operator/(const Quat& p, const Quat& q)
    {
        return p*inv(q);
    }

    //! Returns the product of the specified scalar and quaternion

    friend Quat operator*(const T& s, const Quat& q)
    {
        return Quat(s*q.w,s*q.x,s*q.y,s*q.z);
    }

    //! Returns the product of the specified quaternion and scalar

    friend Quat operator*(const Quat& q, const T& s)
    {
        return Quat(q.w*s,q.x*s,q.y*s,q.z*s);
    }

    //! Returns the quotient of the specified quaternion and scalar

    friend Quat operator/(const Quat& q, const T& s)
    {
        return Quat(q.w/s,q.x/s,q.y/s,q.z/s);
    }

    //! Assigns a quaternion to the sum of the specified quaternions

    friend Quat& operator+=(Quat& p, const Quat& q)
    {
        p.w += q.w;
        p.x += q.x;
        p.y += q.y;
        p.z += q.z;
        return p;
    }

    //! Assigns a quaternion to the difference of the specified quaternions

    friend Quat& operator-=(Quat& p, const Quat& q)
    {
        p.w += q.w;
        p.x += q.x;
        p.y += q.y;
        p.z += q.z;
        return p;
    }

    //! Assigns a quaternion to the product of the specified quaternions

    friend Quat& operator*=(Quat& p, const Quat& q)
    {
        return p = p*q;
    }

    //! Assigns a quaternion to the result of p*inv(q)

    friend Quat& operator/=(Quat& p, const Quat& q)
    {
        return p = p/q;
    }

    //! Assigns a quaternion to the product of the specified quaternion and
    //! scalar

    friend Quat& operator*=(Quat& q, const T& s)
    {
        q.w *= s;
        q.x *= s;
        q.y *= s;
        q.z *= s;
        return q;
    }

    //! Assigns a quaternion to the quotient of the specified quaternion and
    //! scalar

    friend Quat& operator/=(Quat& q, const T& s)
    {
        q.w /= s;
        q.x /= s;
        q.y /= s;
        q.z /= s;
        return q;
    }

    //! Returns a spherical interpolation of the specified quaternions

    friend Quat slerp(const Quat& p, const Quat& q, const T& t)
    {
        T cosRad = dot(unit(p),unit(q));
        T rad = acos(cosRad);
        if (rad < std::numeric_limits<T>::epsilon()) {
            return nlerp(p,q,t);
        }
        else if (cosRad > T(0)) {
            return unit(sin(rad*(T(1)-t))*p+sin(rad*t)*q);
        }
        else {
            rad = M_PI-rad;
            return unit(sin(rad*(T(1)-t))*p-sin(rad*t)*q);
        }
    }

    //! Returns a normalized linear interpolation of the specified quaternions

    friend Quat nlerp(const Quat& p, const Quat& q, const T& t)
    {
        return dot(p,q) > T(0)? unit(p+t*(q-p)): unit(p-t*(q+p));
    }

    //! Returns true if the specified quaternions are equal and false otherwise

    friend bool operator==(const Quat& p, const Quat& q)
    {
        return p.w == q.w && p.x == q.x && p.y == q.y && p.z == q.z;
    }

    //! Returns true if the specified quaternions are not equal and false
    //! otherwise

    friend bool operator!=(const Quat& p, const Quat& q)
    {
        return p.w != q.w || p.x != q.x || p.y != q.y || p.z != q.z;
    }

    //! Writes the elements of the specified quaternion to the output stream

    friend std::ostream& operator<<(std::ostream& os, const Quat& q)
    {
        return os << q.w << " " << q.x << " " << q.y << " " << q.z;
    }

    //! Reads the elements of the specified quaternion from the input stream

    friend std::istream& operator<<(std::istream& is, Quat& q)
    {
        return is >> q.w >> q.x >> q.y >> q.z;
    }

    //! Underlying data array

    union { struct { T w, x, y, z; }; T data[4]; };
};

#endif // QUATERNION_HPP
