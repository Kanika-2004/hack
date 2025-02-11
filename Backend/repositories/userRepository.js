const User=require('../schemas/userSchema')
async function find(parameters){
    try{
        const response=await User.findOne({...parameters});
        console.log("user repo",response)
    return response
    }
    catch(error){
        console.log("a problem occurred in user repository")
        console.log(error);
    }
}
async function create(userdetails){
    try{
        const response=await User.create(userdetails)
        console.log("user repo",response)
        return response
    }catch(error){
        console.log(error);
    }
}
module.exports={
    find,
    create
}