const mongoose=require('mongoose')
const productSchema= new mongoose.Schema({
    name:{
        type: String,
        minlength:[1,"Minimum length should be 5"]
    },
    phone:{
        type:String
    },
    assignedZone:{
        type:String,
        // minlength:[10,"minimum length should be 10 characters"]
    },
    typeofscrap:{
        type:String
    },
    address:{
        type:String
    },
    photo:{
        type: String
    },
    status:{
        type:String,
        // enum:["active","in-active"],
        default:"active"
    },
    safetyGearIssued:{
        type:Boolean,
        default:true
    }
},{timestamps:true})
const product= mongoose.model("product",productSchema)
module.exports=product