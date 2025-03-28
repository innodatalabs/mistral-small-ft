const config = {
	content: [
		'./src/**/*.{html,js,svelte,ts}',
		'./node_modules/flowbite-svelte-icons/**/*.{html,js,svelte,ts}',
	],
	theme: {
		extend: {},
	},
	plugins: [require('@tailwindcss/postcss')],
	// plugins: [require('@tailwindcss/forms')],
};

module.exports = config;
