const express = require("express");
const Startup = require("../schemas/startUp");
const router = express.Router();

// Get all startups
router.get("/", async (req, res) => {
  try {
    const startups = await Startup.find();
    res.json(startups);
  } catch (err) {
    res.status(500).send("Server Error");
  }
});

// Add a new startup
router.post("/", async (req, res) => {
  try {
    const newStartup = new Startup(req.body);
    const savedStartup = await newStartup.save();
    res.json(savedStartup);
  } catch (err) {
    res.status(500).send("Server Error");
  }
});

// Get startup by ID
router.get("/:id", async (req, res) => {
  try {
    const startup = await Startup.findById(req.params.id);
    res.json(startup);
  } catch (err) {
    res.status(500).send("Server Error");
  }
});

// Update a startup
router.put("/:id", async (req, res) => {
  try {
    const updatedStartup = await Startup.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true }
    );
    res.json(updatedStartup);
  } catch (err) {
    res.status(500).send("Server Error");
  }
});

// Delete a startup
router.delete("/:id", async (req, res) => {
  try {
    await Startup.findByIdAndDelete(req.params.id);
    res.json({ message: "Startup deleted successfully" });
  } catch (err) {
    res.status(500).send("Server Error");
  }
});

module.exports = router;
