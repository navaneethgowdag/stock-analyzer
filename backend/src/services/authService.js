const bcrypt = require("bcrypt");
const jwt = require("../utils/jwt");
const userModel = require("../models/userModel");

// ===============================
// REGISTER
// ===============================
exports.register = async (userData) => {

    const { fullName, email, password } = userData;

    // Validation
    if (!fullName || !email || !password) {
        throw new Error("All fields are required");
    }

    // Check if email already exists
    const existingUser = await userModel.findByEmail(email);

    if (existingUser) {
        throw new Error("Email already exists");
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Save user
    const user = await userModel.createUser(
        fullName,
        email,
        hashedPassword
    );

    return {
        success: true,
        message: "Account Created Successfully",
        user: {
            id: user.id,
            full_name: user.full_name,
            email: user.email
        }
    };
};

// ===============================
// LOGIN
// ===============================
exports.login = async (userData) => {

    const { email, password } = userData;

    // Validation
    if (!email || !password) {
        throw new Error("Email and Password are required");
    }

    // Find user
    const user = await userModel.findByEmail(email);

    if (!user) {
        throw new Error("Invalid Email or Password");
    }

    // Compare password
    const isMatch = await bcrypt.compare(
        password,
        user.password_hash
    );

    if (!isMatch) {
        throw new Error("Invalid Email or Password");
    }

    // Generate JWT
    const token = jwt.generateToken(user);

    return {

        success: true,

        message: "Login Successful",

        token,

        user: {

            id: user.id,

            full_name: user.full_name,

            email: user.email

        }

    };

};