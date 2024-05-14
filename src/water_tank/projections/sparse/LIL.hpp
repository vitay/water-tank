#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>

class LILMatrix {
public:
    const unsigned long nb_post;
    const unsigned long nb_pre;

    std::vector<std::vector<int> > ranks; // pre indices sorted by rows
    std::vector<std::vector<float> > values; // values

public:

    LILMatrix(const unsigned long nb_post, const unsigned long nb_pre):
        nb_post(nb_post), nb_pre(nb_pre) {

            for(unsigned int idx=0; idx < this->nb_post; idx++){
                
                this->ranks.push_back(std::vector<int>());

                this->values.push_back(std::vector<float>());
            
            }

    };


    ~LILMatrix() {
        this->ranks.clear();
        this->values.clear();
    }

    unsigned long get_size() {
        unsigned long size = 0;

        for(unsigned int idx=0; idx < this->nb_post; idx++){
            size += this->ranks[idx].size();
        }

        return size;
    }

    void fill_row(int idx, std::vector<int> ranks, std::vector<float> values){

        // Both arrays must be the same size
        if(ranks.size() != values.size()){
            std::cout << "LILMatrix::add_row(): ranks and values must have the same size." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Check that not more values than possible are given.
        if(ranks.size() > this->nb_pre){
            std::cout << "LILMatrix::add_row(): The projection has only " << this->nb_pre << "pre-synaptic neurons, the provided arrays are too long." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Append
        this->ranks[idx] = ranks;
        this->values[idx] = values;
    };

    LILMatrix* uniform_copy(float value) {
        LILMatrix* mat = new LILMatrix(this->nb_post, this->nb_pre);

        int nb_weights;
        for(unsigned int idx=0; idx < this->nb_post; idx++){
            nb_weights = this->ranks[idx].size();
            mat->ranks[idx] = this->ranks[idx];
            mat->values[idx] = std::vector<float>(nb_weights, value);
        }

        return mat; 
    };

    LILMatrix* add_scalar_copy(float value) {
        LILMatrix* mat = new LILMatrix(this->nb_post, this->nb_pre);

        unsigned int nb_weights;
        std::vector<float> vals;
        
        for(unsigned int idx_post=0; idx_post < this->nb_post; idx_post++){
            
            nb_weights = this->ranks[idx_post].size();
            
            mat->ranks[idx_post] = this->ranks[idx_post];
            
            vals = std::vector<float>(nb_weights, 0.0);
        
            for(unsigned int idx_pre=0; idx_pre < nb_weights; idx_pre++){
                vals[idx_pre] = value + this->values[idx_post][idx_pre];
            }
            
            mat->values[idx_post] = vals;
        }

        return mat; 
    };

    void add_scalar_inplace(float value) {

        for(unsigned int idx_post=0; idx_post < this->nb_post; idx_post++){
        
            for(unsigned int idx_pre=0; idx_pre < this->ranks[idx_post].size(); idx_pre++){
                this->values[idx_post][idx_pre] += value;
            }
        }
    };

    LILMatrix* multiply_scalar_copy(float value) {
       
        LILMatrix* mat = new LILMatrix(this->nb_post, this->nb_pre);

        unsigned int nb_weights;
        std::vector<float> vals;
        
        for(unsigned int idx_post=0; idx_post < this->nb_post; idx_post++){
            
            nb_weights = this->ranks[idx_post].size();
            
            mat->ranks[idx_post] = this->ranks[idx_post];
        
            vals = std::vector<float>(nb_weights, 0.0);
        
            for(unsigned int idx_pre=0; idx_pre < nb_weights; idx_pre++){
                vals[idx_pre] = value * this->values[idx_post][idx_pre];
            }
            
            mat->values[idx_post] = vals;
        }

        return mat; 
    };

    void multiply_scalar_inplace(float value) {

        for(unsigned int idx_post=0; idx_post < this->nb_post; idx_post++){
        
            for(unsigned int idx_pre=0; idx_pre < this->ranks[idx_post].size(); idx_pre++){
                this->values[idx_post][idx_pre] *= value;
            }
        }
    };

    std::vector<float> multiply_vector_copy(std::vector<float> vec) {

        float val;
        std::vector<float> res = std::vector<float>(this->nb_post, 0.0);

        for(unsigned int idx_post=0; idx_post < this->nb_post; idx_post++){
        
            val = 0.0;

            for(unsigned int idx_pre=0; idx_pre < this->ranks[idx_post].size(); idx_pre++){
                
                val += this->values[idx_post][idx_pre] * vec[this->ranks[idx_post][idx_pre]];
            
            }
            
            res[idx_post] = val;
        }
        

        return res; 
    };



    void add_matrix_inplace(LILMatrix *other) {

        for(unsigned int idx_post=0; idx_post < this->nb_post; idx_post++){
        
            for(unsigned int idx_pre=0; idx_pre < this->ranks[idx_post].size(); idx_pre++){
                
                this->values[idx_post][idx_pre] += other->values[idx_post][idx_pre];
            }
        }
    };


    void substract_matrix_inplace(LILMatrix *other) {

        for(unsigned int idx_post=0; idx_post < this->nb_post; idx_post++){
        
            for(unsigned int idx_pre=0; idx_pre < this->ranks[idx_post].size(); idx_pre++){
                
                this->values[idx_post][idx_pre] -= other->values[idx_post][idx_pre];
            }
        }
    };

    LILMatrix* outer_product(std::vector<float> left, std::vector<float> right){

        LILMatrix* mat = new LILMatrix(this->nb_post, this->nb_pre);

        unsigned int nb_weights;
        std::vector<float> vals;

        for(unsigned int idx_post=0; idx_post < this->nb_post; idx_post++){

            nb_weights = this->ranks[idx_post].size();
            
            mat->ranks[idx_post] = this->ranks[idx_post];
        
            vals = std::vector<float>(nb_weights, 0.0);
        
            for(unsigned int idx_pre=0; idx_pre < this->ranks[idx_post].size(); idx_pre++){
                
                vals[idx_pre] = left[idx_post] * right[this->ranks[idx_post][idx_pre]];
            }

            mat->values[idx_post] = vals;
        }

        return mat;
    };
}; 