const cloudinary=require('../config/cloudinaryConfig')
const fs=require('fs/promises')
const productRepository=require('../repositories/scrapRepo')
async function findproductbyid(productid){
    const response=await productRepository.findProductById(productid);
    console.log(response)
    if(!response){
        throw{message:"no product found"}
    }
    return response
}
async function findproduct(){
    const response=await productRepository.findallProduct();
    console.log(response)
    if(!response){
        throw{message:"no product found"}
    
    }
    return response
}
async function deleteproductbyid(productid){
    const response=await productRepository.deleteProductById(productid);
    console.log(response)
    if(!response){
        throw{message:"cannot delete the product"}
    }
    return response
}
async function createproduct(productdetails){
    const imagepath=productdetails.photo
    if(imagepath){
        try{
        const resp=await cloudinary.uploader.upload(imagepath)
        var productImag=resp.secure_url
        console.log(productImag)
        await fs.unlink(process.cwd() + "/" + imagepath);
        }catch(error){
            console.log(error)
            throw{message:"error occurred while uploading"}
        }
    }
    const response=await productRepository.createProduct({
        ...productdetails,
        photo:productImag
    });
    console.log("AT product serviice  ",response)
    if(!response){
        throw{message:"no product found"}
    }

    return response
}
module.exports={
    createproduct,
    findproduct,
    findproductbyid,
    deleteproductbyid
}