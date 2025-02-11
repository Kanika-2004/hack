const registeruser=require('../services/userService')
async function createuser(req,res){
    try{
        const response=await registeruser(req.body);
        console.log("user controller",response)
        return res.status(201).json({
            success:true,
            error:{},
            data:response,
            message:"successfully registered the user"
        })
    }catch(error){
        console.log("user controller",error);
        return res.status(404).json({
            success:false,
            error:error,
            message:"unable to create user"
        })
    }
}
module.exports=createuser