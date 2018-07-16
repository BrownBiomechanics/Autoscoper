// Filename: cParticle.h

#ifndef C_PARTICLE_H
#define C_PARTICLE_H

const int MAX_INPUTS = 6;

class cParticle
{

private:
	double     mData[MAX_INPUTS];
	double     mpBest;
	float   mVelocity;

public:
	cParticle();
	double      getData(int index) const;
	void     setData(int index, double value);
	double      getpBest() const;
	void     setpBest(double value);
	float    getVelocity() const;
	void     setVelocity(float value);

}; // end cParticle class.

#endif
