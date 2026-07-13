const pool = require("../config/db");

exports.findByEmail = async (email) => {

    console.log("Searching for:", email);

    const result = await pool.query(

        "SELECT * FROM users WHERE email = $1",

        [email]

    );

    console.log(result.rows);

    return result.rows[0];

};

exports.createUser = async (name, email, passwordHash) => {

    const result = await pool.query(

        `

        INSERT INTO users
        (full_name,email,password_hash)

        VALUES($1,$2,$3)

        RETURNING *

        `,

        [

            name,

            email,

            passwordHash

        ]

    );

    return result.rows[0];

};
