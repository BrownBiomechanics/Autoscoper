// Filename: cParticle.cpp

#include "cParticle.h"

cParticle::cParticle()
{
	mpBest = 0.0;
	mVelocity = 0.0;
}

double cParticle::getData(int index) const
{
	return this->mData[index];
}

void cParticle::setData(int index, double value)
{
	this->mData[index] = value;
}

double cParticle::getpBest() const
{
	return this->mpBest;
}

void cParticle::setpBest(double value)
{
	this->mpBest = value;
}

float cParticle::getVelocity() const
{
	return this->mVelocity;
}

void cParticle::setVelocity(float value)
{
	this->mVelocity = value;
}
