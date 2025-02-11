const {createproduct,findproduct,findproductbyid,deleteproductbyid}=require('../services/productService')
async function createProduct(req,res){
    try{
    const response=await createproduct({
        name:req.body.name,
        photo:req.file?.path,
        assignedZone:req.body.assignedZone,
        safetyGearIssued:req.body.safetyGearIssued,
        address:req.body.address,
        phone:req.body.phone,
        typeofscrap:req.body.typeofscrap

    })
    console.log("Product Controller results: ",response)
    return res.status(201).json({
        success:true,
        data:response,
        error:{},
        message:"successfully registered"
    })
}catch(error){
    console.log(error)
    return res.status(500).json({
        success:{},
        error:error,
        message:"could not create product"
    })
    throw{message:"could not create product"}
}
}
async function findproductbyId(req,res){
    try{
    const response= await findproductbyid(req.params.id)
    console.log(response)
        return res.status(201).json({
            success:true,
            error:{},
            data:response,
            message:"found the product"
        })
}catch(error){
    console.log(error)
    return res.status(500).json({
        success:false,
        error:error,
        message:"could not find the product"
    })
}
}
async function findproducts(req,res){
    try{
    const response= await findproduct();
    if(!response){
        return res.status(404).json({
            message:"could not find the products"
        })
    }
    if(response){ 
    return res.status(201).json({
            success:true,
            error:{},
            data:response,
            message:"found the products"
        })
    }
}catch(error){
    console.log(error)
    return res.status(500).json({
        success:false,
        error:error,
        message:"could not find the product"
    })
}
}
async function deleteProduct(req,res){
    try{
    const response= await deleteproductbyid(req.params.id)
        return res.status(201).json({
            success:true,
            error:{},
            data:response,
            message:" deleted the product"
        })
}catch(error){
    console.log(error)
    return res.status(500).json({
        success:false,
        error:error,
        message:"could not delete the product"
    })
}
}
module.exports={
    findproducts,
    findproductbyId,
    deleteProduct,
    createProduct
}