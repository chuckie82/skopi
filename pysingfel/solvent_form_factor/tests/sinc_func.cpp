class SincFunction :

 // Constructor
   def __init__(max_value, float bin_size):
      bin_size_ = bin_size;
      one_over_bin_size_ = 1.0 / bin_size_
      max_value_ = max_value;
   unsigned int size = value2index(max_value_) + 1;
   reserve(size);
   for (unsigned int i = 0; i <= size; i++) {
     float x = i * bin_size_;
     push_back(boost::math::sinc_pi(x));
   }
 }

 unsigned int value2index(float value) const {
   return IMP::algebra::get_rounded(value * one_over_bin_size_);
 }

 // get sinc value for x, compute values if they weren't computed yet
 float sinc(float x) {
   unsigned int index = value2index(x);
   if (index >= size()) {
     reserve(index);
     for (unsigned int i = size(); i <= index; i++) {
       float x = i * bin_size_;
       push_back(boost::math::sinc_pi(x));
     }
   }
   return (*this)[index];
 }
