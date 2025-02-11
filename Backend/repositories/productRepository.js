const product=require('../schemas/productSchema')
const mongoose=require('mongoose')
async function createProduct(productdetails){
    try{
        const response=await product.create(productdetails)
        return response
    }catch(error){
        console.log(error)
        throw{message:"unable to create product"}
    }
}
async function findProductById(productid){
    try{
        
        const response=await product.findById(productid)
        return response
    }catch(error){
        console.log(error)
        throw{message:"unable to find product"}
    }
}
async function findallProduct(){
    try{
        const response=await product.find()
        return response
    }catch(error){
        console.log(error)
        throw{message:"unable to find all product"}
    }
}
async function deleteProductById(productid){
    try{
        const response=await product.findByIdAndDelete(productid)
        return response
    }catch(error){
        console.log(error)
        throw{message:"unable to delete product"}
    }
}
module.exports={
    createProduct,
    findProductById,
    findallProduct,
    deleteProductById
}